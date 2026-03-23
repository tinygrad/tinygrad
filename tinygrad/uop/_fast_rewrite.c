#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* C implementation of the unified_rewrite inner loop.
 * Inlines PatternMatcher.rewrite dispatch + bpm_cache lookup to eliminate Python callback overhead.
 * Match functions (compiled pattern matchers) are still called in Python. */

static PyObject *str_op = NULL, *str_src = NULL, *str_dtype = NULL, *str_arg = NULL, *str_tag = NULL;
static PyObject *str_src_ops = NULL, *str_issubset = NULL;
static PyObject *OPS_CALL = NULL, *BottomUpGate = NULL, *UOp_class = NULL;

static int init_constants(void) {
    if (str_op != NULL) return 0;
    str_op = PyUnicode_InternFromString("op");
    str_src = PyUnicode_InternFromString("src");
    str_dtype = PyUnicode_InternFromString("dtype");
    str_arg = PyUnicode_InternFromString("arg");
    str_tag = PyUnicode_InternFromString("tag");
    str_src_ops = PyUnicode_InternFromString("_src_ops");
    str_issubset = PyUnicode_InternFromString("issubset");
    if (!str_op || !str_src || !str_dtype || !str_arg || !str_tag || !str_src_ops || !str_issubset) return -1;

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

/* === Stack === */
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

static int waitlist_add(PyObject *waitlist, PyObject *key, PyObject *n, int stage, PyObject *new_n) {
    PyObject *wl = PyDict_GetItem(waitlist, key);
    if (!wl) {
        wl = PyList_New(0);
        if (!wl) return -1;
        if (PyDict_SetItem(waitlist, key, wl) < 0) { Py_DECREF(wl); return -1; }
        Py_DECREF(wl);
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

static int flush_waitlist(Stack *s, PyObject *waitlist, PyObject *key) {
    PyObject *wl = PyDict_GetItem(waitlist, key);
    if (wl) {
        if (stack_extend_from_list(s, wl) < 0) return -1;
        PyDict_DelItem(waitlist, key);
    }
    return 0;
}

/* === Inline PatternMatcher.rewrite ===
 * Returns new ref: UOp (match), Py_None (no match), or NULL (error/exception).
 * Match functions may raise BottomUpGate. */
static PyObject* pm_dispatch(PyObject *pdict, PyObject *uop, PyObject *ctx) {
    PyObject *op = PyObject_GetAttr(uop, str_op);
    if (!op) return NULL;
    PyObject *pats = PyDict_GetItem(pdict, op);  /* borrowed */
    Py_DECREF(op);
    if (!pats || PyList_GET_SIZE(pats) == 0) Py_RETURN_NONE;

    /* Get or compute _src_ops = {u.op for u in uop.src} */
    PyObject *inst_dict = PyObject_GenericGetDict(uop, NULL);
    if (!inst_dict) return NULL;
    PyObject *ler = PyDict_GetItem(inst_dict, str_src_ops);  /* borrowed */
    if (!ler) {
        PyObject *src = PyObject_GetAttr(uop, str_src);
        if (!src) { Py_DECREF(inst_dict); return NULL; }
        ler = PySet_New(NULL);
        if (!ler) { Py_DECREF(src); Py_DECREF(inst_dict); return NULL; }
        Py_ssize_t slen = PyTuple_GET_SIZE(src);
        for (Py_ssize_t i = 0; i < slen; i++) {
            PyObject *child_op = PyObject_GetAttr(PyTuple_GET_ITEM(src, i), str_op);
            if (!child_op) { Py_DECREF(ler); Py_DECREF(src); Py_DECREF(inst_dict); return NULL; }
            if (PySet_Add(ler, child_op) < 0) { Py_DECREF(child_op); Py_DECREF(ler); Py_DECREF(src); Py_DECREF(inst_dict); return NULL; }
            Py_DECREF(child_op);
        }
        Py_DECREF(src);
        if (PyDict_SetItem(inst_dict, str_src_ops, ler) < 0) { Py_DECREF(ler); Py_DECREF(inst_dict); return NULL; }
        Py_DECREF(ler);
        ler = PyDict_GetItem(inst_dict, str_src_ops);
    }
    Py_DECREF(inst_dict);

    /* Check patterns: entry = [UPat, match_fn, early_reject_frozenset] */
    Py_ssize_t npats = PyList_GET_SIZE(pats);
    for (Py_ssize_t j = 0; j < npats; j++) {
        PyObject *entry = PyList_GET_ITEM(pats, j);
        PyObject *match_fn = PyList_GET_ITEM(entry, 1);
        PyObject *early_reject = PyList_GET_ITEM(entry, 2);

        /* early_reject.issubset(ler) */
        PyObject *is_sub = PyObject_CallMethodOneArg(early_reject, str_issubset, ler);
        if (!is_sub) return NULL;
        int pass = PyObject_IsTrue(is_sub);
        Py_DECREF(is_sub);
        if (pass < 0) return NULL;
        if (!pass) continue;

        /* match(uop, ctx) */
        PyObject *result = PyObject_CallFunctionObjArgs(match_fn, uop, ctx, NULL);
        if (!result) return NULL;
        if (result != Py_None && result != uop) return result;
        Py_DECREF(result);
    }
    Py_RETURN_NONE;
}

/*
 * c_unified_rewrite(root, bpm_cache, bpm_pdict, pm_pdict, ctx,
 *                   enter_calls, replace, bpm_is_none, limit) -> UOp
 */
static PyObject* c_unified_rewrite(PyObject *self, PyObject *args) {
    PyObject *root, *bpm_cache, *bpm_pdict, *pm_pdict, *ctx, *replace;
    int enter_calls, bpm_is_none;
    long limit;

    if (!PyArg_ParseTuple(args, "OOOOOiOil",
            &root, &bpm_cache, &bpm_pdict, &pm_pdict, &ctx,
            &enter_calls, &replace, &bpm_is_none, &limit))
        return NULL;
    if (init_constants() < 0) return NULL;

    Stack stack;
    if (stack_init(&stack, 4096) < 0) { PyErr_NoMemory(); return NULL; }

    PyObject *on_stack = PySet_New(NULL);
    PyObject *waitlist = PyDict_New();
    PyObject *refs_list = PyList_New(0);
    if (!on_stack || !waitlist || !refs_list) goto error;

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

        if (PyDict_Contains(replace, n)) continue;

        if (stage == 0) {
            /* === Bottom-up rewrite (inline cached_bpm_rewrite + pm_dispatch) === */
            if (!bpm_is_none) {
                PyObject *test_n = n;
                int got_gate = 0;

                /* Inline cache lookup */
                PyObject *cached = PyDict_GetItem(bpm_cache, n);  /* borrowed, NULL=miss */
                PyObject *first;
                if (cached) {
                    first = cached;
                    Py_INCREF(first);
                } else {
                    /* Cache miss: dispatch inline */
                    first = pm_dispatch(bpm_pdict, n, ctx);
                    if (!first) {
                        if (PyErr_ExceptionMatches(BottomUpGate)) {
                            PyErr_Clear();
                            /* BottomUpGate: don't cache, set replace[n]=n */
                            if (PyDict_SetItem(replace, n, n) < 0) goto error;
                            if (flush_waitlist(&stack, waitlist, n) < 0) goto error;
                            continue;
                        }
                        goto error;
                    }
                    /* Store in cache */
                    if (PyDict_SetItem(bpm_cache, n, first) < 0) { Py_DECREF(first); goto error; }
                }

                if (first != Py_None) {
                    new_n = first;
                    test_n = first;
                    if (PyList_Append(refs_list, first) < 0) { Py_DECREF(first); goto error; }
                    Py_DECREF(first);

                    /* Fixed-point iteration */
                    PyObject *seen = PySet_New(NULL);
                    if (!seen) goto error;
                    PySet_Add(seen, n);
                    PySet_Add(seen, new_n);

                    for (;;) {
                        /* Inline cache lookup for fixed-point */
                        PyObject *fp_cached = PyDict_GetItem(bpm_cache, test_n);
                        PyObject *next;
                        if (fp_cached) {
                            next = fp_cached;
                            Py_INCREF(next);
                        } else {
                            next = pm_dispatch(bpm_pdict, test_n, ctx);
                            if (!next) {
                                if (PyErr_ExceptionMatches(BottomUpGate)) {
                                    PyErr_Clear();
                                    if (PyDict_SetItem(replace, n, test_n) < 0) { Py_DECREF(seen); goto error; }
                                    if (flush_waitlist(&stack, waitlist, n) < 0) { Py_DECREF(seen); goto error; }
                                    Py_DECREF(seen);
                                    got_gate = 1;
                                    break;
                                }
                                Py_DECREF(seen); goto error;
                            }
                            if (PyDict_SetItem(bpm_cache, test_n, next) < 0) { Py_DECREF(next); Py_DECREF(seen); goto error; }
                        }

                        if (next == Py_None) {
                            Py_DECREF(next);
                            break;
                        }

                        if (PySet_Contains(seen, next)) {
                            Py_DECREF(seen); Py_DECREF(next);
                            PyErr_SetString(PyExc_RuntimeError, "infinite loop in fixed_point_rewrite");
                            goto error;
                        }
                        PySet_Add(seen, next);
                        new_n = next;
                        test_n = next;
                        if (PyList_Append(refs_list, next) < 0) { Py_DECREF(next); Py_DECREF(seen); goto error; }
                        Py_DECREF(next);
                    }
                    if (!got_gate) Py_DECREF(seen);
                    if (got_gate) continue;
                } else {
                    Py_DECREF(first);
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
                    if (waitlist_add(waitlist, x, n, 1, new_n) < 0) { Py_DECREF(src); goto error; }
                    all_ready = 0;
                    break;
                }
                if (rx != x) any_changed = 1;
            }
            Py_DECREF(src);
            if (!all_ready) continue;

            if (!any_changed) {
                /* No src changes: try pm_dispatch inline if op has patterns */
                int skip_pm = 1;
                if (pm_pdict != Py_None) {
                    PyObject *op = PyObject_GetAttr(new_n, str_op);
                    if (!op) goto error;
                    int has = PyDict_Contains(pm_pdict, op);
                    Py_DECREF(op);
                    if (has < 0) goto error;
                    if (has) {
                        PyObject *rw = pm_dispatch(pm_pdict, new_n, ctx);
                        if (!rw) goto error;
                        if (rw != Py_None) {
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

                if (PyList_Append(refs_list, new_uop) < 0) { Py_DECREF(new_uop); goto error; }
                STACK_PUSH(&stack, n, 2, new_uop);
                STACK_PUSH(&stack, new_uop, 0, new_uop);
                Py_DECREF(new_uop);
            }

        } else {
            /* === Stage 2: link result === */
            PyObject *rep = PyDict_GetItemWithError(replace, new_n);
            if (!rep) {
                if (PyErr_Occurred()) goto error;
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
    {"c_unified_rewrite", c_unified_rewrite, METH_VARARGS, "C unified_rewrite with inline PM dispatch"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_fast_rewrite", NULL, -1, methods
};

PyMODINIT_FUNC PyInit__fast_rewrite(void) {
    return PyModule_Create(&module);
}
