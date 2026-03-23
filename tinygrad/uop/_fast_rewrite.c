#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* C implementation of the unified_rewrite inner loop.
 * This eliminates Python bytecode overhead for the hot loop that dominates graph_rewrite. */

static PyObject *str_op = NULL, *str_src = NULL, *str_dtype = NULL, *str_arg = NULL, *str_tag = NULL;
static PyObject *OPS_CALL = NULL, *BottomUpGate = NULL, *UOp_class = NULL;

static int init_constants(void) {
    if (str_op != NULL) return 0;
    str_op = PyUnicode_InternFromString("op");
    str_src = PyUnicode_InternFromString("src");
    str_dtype = PyUnicode_InternFromString("dtype");
    str_arg = PyUnicode_InternFromString("arg");
    str_tag = PyUnicode_InternFromString("tag");
    if (!str_op || !str_src || !str_dtype || !str_arg || !str_tag) return -1;

    PyObject *ops_mod = PyImport_ImportModule("tinygrad.uop.ops");
    if (!ops_mod) return -1;
    BottomUpGate = PyObject_GetAttrString(ops_mod, "BottomUpGate");
    UOp_class = PyObject_GetAttrString(ops_mod, "UOp");
    PyObject *ops_enum = PyObject_GetAttrString(ops_mod, "Ops");
    if (ops_enum) { OPS_CALL = PyObject_GetAttrString(ops_enum, "CALL"); Py_DECREF(ops_enum); }
    Py_DECREF(ops_mod);
    if (!BottomUpGate || !OPS_CALL || !UOp_class) return -1;
    return 0;
}

/* Stack using parallel arrays with borrowed references.
 * A separate refs_list (Python list) holds strong references to keep objects alive. */
typedef struct {
    PyObject **n;
    int *stage;
    PyObject **new_n;
    Py_ssize_t size;
    Py_ssize_t cap;
} Stack;

static int stack_init(Stack *s, Py_ssize_t cap) {
    s->n = (PyObject**)PyMem_Malloc(cap * sizeof(PyObject*));
    s->stage = (int*)PyMem_Malloc(cap * sizeof(int));
    s->new_n = (PyObject**)PyMem_Malloc(cap * sizeof(PyObject*));
    s->size = 0; s->cap = cap;
    return (s->n && s->stage && s->new_n) ? 0 : -1;
}
static void stack_free(Stack *s) {
    PyMem_Free(s->n); PyMem_Free(s->stage); PyMem_Free(s->new_n);
}
static int stack_grow(Stack *s) {
    Py_ssize_t nc = s->cap * 2;
    PyObject **tn = (PyObject**)PyMem_Realloc(s->n, nc * sizeof(PyObject*));
    int *ts = (int*)PyMem_Realloc(s->stage, nc * sizeof(int));
    PyObject **tnn = (PyObject**)PyMem_Realloc(s->new_n, nc * sizeof(PyObject*));
    if (!tn || !ts || !tnn) return -1;
    s->n = tn; s->stage = ts; s->new_n = tnn; s->cap = nc;
    return 0;
}

#define STACK_PUSH(s, _n, _st, _nn) do { \
    if ((s)->size >= (s)->cap && stack_grow(s) < 0) goto error; \
    (s)->n[(s)->size] = (_n); (s)->stage[(s)->size] = (_st); (s)->new_n[(s)->size] = (_nn); (s)->size++; \
} while(0)

/* Extend stack from a Python list of (n, stage_int, new_n) tuples */
static int stack_extend_from_list(Stack *s, PyObject *lst) {
    Py_ssize_t len = PyList_GET_SIZE(lst);
    for (Py_ssize_t i = 0; i < len; i++) {
        PyObject *tup = PyList_GET_ITEM(lst, i);
        PyObject *_n = PyTuple_GET_ITEM(tup, 0);
        int _st = (int)PyLong_AsLong(PyTuple_GET_ITEM(tup, 1));
        PyObject *_nn = PyTuple_GET_ITEM(tup, 2);
        if (s->size >= s->cap && stack_grow(s) < 0) return -1;
        s->n[s->size] = _n; s->stage[s->size] = _st; s->new_n[s->size] = _nn; s->size++;
    }
    return 0;
}

/* Helper: add (n, stage, new_n) tuple to a waitlist entry */
static int waitlist_add(PyObject *waitlist, PyObject *key, PyObject *n, int stage, PyObject *new_n) {
    PyObject *wl = PyDict_GetItem(waitlist, key);
    if (!wl) {
        wl = PyList_New(0);
        if (!wl) return -1;
        if (PyDict_SetItem(waitlist, key, wl) < 0) { Py_DECREF(wl); return -1; }
        Py_DECREF(wl);  /* dict holds the ref now */
    }
    PyObject *stage_obj = PyLong_FromLong(stage);
    if (!stage_obj) return -1;
    PyObject *entry = PyTuple_Pack(3, n, stage_obj, new_n);
    Py_DECREF(stage_obj);
    if (!entry) return -1;
    int rc = PyList_Append(wl, entry);
    Py_DECREF(entry);
    return rc;
}

/* Helper: if key in waitlist, extend stack from its entries and delete key */
static int flush_waitlist(Stack *s, PyObject *waitlist, PyObject *key) {
    PyObject *wl = PyDict_GetItem(waitlist, key);
    if (wl) {
        if (stack_extend_from_list(s, wl) < 0) return -1;
        PyDict_DelItem(waitlist, key);
    }
    return 0;
}

/*
 * c_unified_rewrite(root, cached_bpm_rewrite, pm_rewrite, pm_pdict,
 *                   enter_calls, replace, bpm_is_none, limit) -> UOp
 */
static PyObject* c_unified_rewrite(PyObject *self, PyObject *args) {
    PyObject *root, *cached_bpm_rewrite, *pm_rewrite_fn, *pm_pdict, *replace;
    int enter_calls, bpm_is_none;
    long limit;

    if (!PyArg_ParseTuple(args, "OOOOiOil",
            &root, &cached_bpm_rewrite, &pm_rewrite_fn, &pm_pdict,
            &enter_calls, &replace, &bpm_is_none, &limit))
        return NULL;
    if (init_constants() < 0) return NULL;

    Stack stack;
    if (stack_init(&stack, 4096) < 0) { PyErr_NoMemory(); return NULL; }

    PyObject *on_stack = PySet_New(NULL);
    PyObject *waitlist = PyDict_New();
    /* refs_list keeps strong references to dynamically created UOps that the stack references */
    PyObject *refs_list = PyList_New(0);
    if (!on_stack || !waitlist || !refs_list) goto error;

    /* Push root */
    STACK_PUSH(&stack, root, 0, root);
    if (PySet_Add(on_stack, root) < 0) goto error;

    while (stack.size > 0) {
        if (stack.size > limit) {
            PyErr_SetString(PyExc_RuntimeError, "infinite loop in graph_rewrite (stack too big)");
            goto error;
        }

        stack.size--;
        PyObject *n = stack.n[stack.size];
        int stage = stack.stage[stack.size];
        PyObject *new_n = stack.new_n[stack.size];

        /* if n in replace: continue */
        if (PyDict_Contains(replace, n)) continue;

        if (stage == 0) {
            /* === Bottom-up rewrite === */
            if (!bpm_is_none) {
                PyObject *test_n = n;
                int got_gate = 0;

                PyObject *first = PyObject_CallOneArg(cached_bpm_rewrite, n);
                if (!first) {
                    if (PyErr_ExceptionMatches(BottomUpGate)) {
                        PyErr_Clear();
                        if (PyDict_SetItem(replace, n, n) < 0) goto error;
                        if (flush_waitlist(&stack, waitlist, n) < 0) goto error;
                        continue;
                    }
                    goto error;
                }

                if (first != Py_None) {
                    new_n = first;
                    test_n = first;
                    /* Keep first alive: add to refs_list */
                    if (PyList_Append(refs_list, first) < 0) { Py_DECREF(first); goto error; }
                    Py_DECREF(first);  /* refs_list now holds the ref */

                    PyObject *seen = PySet_New(NULL);
                    if (!seen) goto error;
                    PySet_Add(seen, n);
                    PySet_Add(seen, first);

                    PyObject *next = PyObject_CallOneArg(cached_bpm_rewrite, first);
                    if (!next) {
                        if (PyErr_ExceptionMatches(BottomUpGate)) {
                            PyErr_Clear();
                            if (test_n == Py_None) { Py_DECREF(seen); PyErr_SetString(PyExc_RuntimeError, "unwrap None"); goto error; }
                            if (PyDict_SetItem(replace, n, test_n) < 0) { Py_DECREF(seen); goto error; }
                            if (flush_waitlist(&stack, waitlist, n) < 0) { Py_DECREF(seen); goto error; }
                            Py_DECREF(seen);
                            continue;
                        }
                        Py_DECREF(seen); goto error;
                    }

                    while (next != Py_None) {
                        if (PySet_Contains(seen, next)) {
                            Py_DECREF(seen); Py_DECREF(next);
                            PyErr_SetString(PyExc_RuntimeError, "infinite loop in fixed_point_rewrite");
                            goto error;
                        }
                        PySet_Add(seen, next);
                        new_n = next;
                        test_n = next;
                        /* Keep alive */
                        if (PyList_Append(refs_list, next) < 0) { Py_DECREF(next); Py_DECREF(seen); goto error; }
                        Py_DECREF(next);

                        next = PyObject_CallOneArg(cached_bpm_rewrite, next);
                        if (!next) {
                            if (PyErr_ExceptionMatches(BottomUpGate)) {
                                PyErr_Clear();
                                if (test_n == Py_None) { Py_DECREF(seen); PyErr_SetString(PyExc_RuntimeError, "unwrap None"); goto error; }
                                if (PyDict_SetItem(replace, n, test_n) < 0) { Py_DECREF(seen); goto error; }
                                if (flush_waitlist(&stack, waitlist, n) < 0) { Py_DECREF(seen); goto error; }
                                Py_DECREF(seen);
                                got_gate = 1;
                                break;
                            }
                            Py_DECREF(seen); goto error;
                        }
                    }
                    if (!got_gate) {
                        Py_DECREF(seen);
                        if (next == Py_None) Py_DECREF(next);
                    }
                    if (got_gate) continue;
                } else {
                    Py_DECREF(first);  /* first == Py_None */
                }
            }

            /* Push (n, 1, new_n) */
            STACK_PUSH(&stack, n, 1, new_n);

            /* CALL handling */
            if (!enter_calls) {
                PyObject *op = PyObject_GetAttr(new_n, str_op);
                if (!op) goto error;
                int is_call = (op == OPS_CALL);
                Py_DECREF(op);
                if (is_call) {
                    PyObject *src = PyObject_GetAttr(new_n, str_src);
                    if (!src) goto error;
                    PyObject *src0 = PyTuple_GET_ITEM(src, 0);
                    PyDict_SetItem(replace, src0, src0);
                    Py_DECREF(src);
                }
            }

            /* Push children in reverse */
            {
                PyObject *src = PyObject_GetAttr(new_n, str_src);
                if (!src) goto error;
                Py_ssize_t slen = PyTuple_GET_SIZE(src);
                for (Py_ssize_t i = slen - 1; i >= 0; i--) {
                    PyObject *x = PyTuple_GET_ITEM(src, i);
                    int in_set = PySet_Contains(on_stack, x);
                    if (in_set < 0) { Py_DECREF(src); goto error; }
                    if (!in_set) {
                        STACK_PUSH(&stack, x, 0, x);
                        if (PySet_Add(on_stack, x) < 0) { Py_DECREF(src); goto error; }
                    }
                }
                Py_DECREF(src);
            }

        } else if (stage == 1) {
            /* === Check srcs ready + rebuild === */
            int any_changed = 0, all_ready = 1;
            PyObject *src = PyObject_GetAttr(new_n, str_src);
            if (!src) goto error;
            Py_ssize_t slen = PyTuple_GET_SIZE(src);

            for (Py_ssize_t i = 0; i < slen; i++) {
                PyObject *x = PyTuple_GET_ITEM(src, i);
                PyObject *rx = PyDict_GetItemWithError(replace, x);
                if (!rx) {
                    if (PyErr_Occurred()) { Py_DECREF(src); goto error; }
                    /* Not ready: add to waitlist */
                    if (waitlist_add(waitlist, x, n, 1, new_n) < 0) { Py_DECREF(src); goto error; }
                    all_ready = 0;
                    break;
                }
                if (rx != x) any_changed = 1;
            }
            Py_DECREF(src);
            if (!all_ready) continue;

            if (!any_changed) {
                /* No src changes: try pm_rewrite if op has patterns */
                int skip_pm = 1;
                if (pm_pdict != Py_None) {
                    PyObject *op = PyObject_GetAttr(new_n, str_op);
                    if (!op) goto error;
                    int has = PyDict_Contains(pm_pdict, op);
                    Py_DECREF(op);
                    if (has < 0) goto error;
                    if (has) {
                        PyObject *rw = PyObject_CallOneArg(pm_rewrite_fn, new_n);
                        if (!rw) goto error;
                        if (rw != Py_None) {
                            /* pm matched: push for re-traverse */
                            if (PyList_Append(refs_list, rw) < 0) { Py_DECREF(rw); goto error; }
                            STACK_PUSH(&stack, n, 2, rw);
                            STACK_PUSH(&stack, rw, 0, rw);
                            Py_DECREF(rw);
                            skip_pm = 0;
                        } else {
                            Py_DECREF(rw);
                        }
                    }
                }
                if (skip_pm) {
                    if (PyDict_SetItem(replace, n, new_n) < 0) goto error;
                    if (flush_waitlist(&stack, waitlist, n) < 0) goto error;
                }
            } else {
                /* Srcs changed: build new UOp */
                PyObject *src2 = PyObject_GetAttr(new_n, str_src);
                if (!src2) goto error;
                Py_ssize_t slen2 = PyTuple_GET_SIZE(src2);
                PyObject *new_src_tuple = PyTuple_New(slen2);
                if (!new_src_tuple) { Py_DECREF(src2); goto error; }
                for (Py_ssize_t i = 0; i < slen2; i++) {
                    PyObject *x = PyTuple_GET_ITEM(src2, i);
                    PyObject *rx = PyDict_GetItem(replace, x);
                    if (!rx) rx = x;
                    Py_INCREF(rx);
                    PyTuple_SET_ITEM(new_src_tuple, i, rx);
                }
                Py_DECREF(src2);

                PyObject *nn_op = PyObject_GetAttr(new_n, str_op);
                PyObject *nn_dt = PyObject_GetAttr(new_n, str_dtype);
                PyObject *nn_arg = PyObject_GetAttr(new_n, str_arg);
                PyObject *nn_tag = PyObject_GetAttr(new_n, str_tag);
                if (!nn_op || !nn_dt || !nn_arg || !nn_tag) {
                    Py_XDECREF(nn_op); Py_XDECREF(nn_dt); Py_XDECREF(nn_arg); Py_XDECREF(nn_tag);
                    Py_DECREF(new_src_tuple); goto error;
                }
                PyObject *new_uop = PyObject_CallFunctionObjArgs(UOp_class, nn_op, nn_dt, new_src_tuple, nn_arg, nn_tag, NULL);
                Py_DECREF(nn_op); Py_DECREF(nn_dt); Py_DECREF(nn_arg); Py_DECREF(nn_tag);
                Py_DECREF(new_src_tuple);
                if (!new_uop) goto error;

                /* Keep alive + push */
                if (PyList_Append(refs_list, new_uop) < 0) { Py_DECREF(new_uop); goto error; }
                STACK_PUSH(&stack, n, 2, new_uop);
                STACK_PUSH(&stack, new_uop, 0, new_uop);
                Py_DECREF(new_uop);  /* refs_list holds it */
            }

        } else {
            /* === Stage 2: link result === */
            PyObject *rep = PyDict_GetItemWithError(replace, new_n);
            if (!rep) {
                if (PyErr_Occurred()) goto error;
                /* Not ready */
                if (waitlist_add(waitlist, new_n, n, 2, new_n) < 0) goto error;
            } else {
                if (PyDict_SetItem(replace, n, rep) < 0) goto error;
                if (flush_waitlist(&stack, waitlist, n) < 0) goto error;
            }
        }
    }

    /* return replace[root] */
    PyObject *result = PyDict_GetItem(replace, root);
    if (!result) {
        PyErr_SetString(PyExc_KeyError, "root not in replace after unified_rewrite");
        goto error;
    }
    Py_INCREF(result);
    stack_free(&stack);
    Py_DECREF(on_stack);
    Py_DECREF(waitlist);
    Py_DECREF(refs_list);
    return result;

error:
    stack_free(&stack);
    Py_XDECREF(on_stack);
    Py_XDECREF(waitlist);
    Py_XDECREF(refs_list);
    return NULL;
}

static PyMethodDef methods[] = {
    {"c_unified_rewrite", c_unified_rewrite, METH_VARARGS, "C unified_rewrite inner loop"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_fast_rewrite", NULL, -1, methods
};

PyMODINIT_FUNC PyInit__fast_rewrite(void) {
    return PyModule_Create(&module);
}
