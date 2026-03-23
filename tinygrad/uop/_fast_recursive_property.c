#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

/* C implementation of recursive_property descriptor.
 * The medium path (check children, compute, cache) is done entirely in C,
 * avoiding Python frame overhead for the ~500K calls in stable_diffusion. */

static PyObject *str_src = NULL;

static int init_strs(void) {
    if (str_src) return 0;
    str_src = PyUnicode_InternFromString("src");
    return str_src ? 0 : -1;
}

typedef struct {
    PyObject_HEAD
    PyObject *fxn;      /* the wrapped Python function */
    PyObject *nm;       /* property name (interned string) */
    PyObject *doc;      /* __doc__ */
} RecPropObject;

static void recprop_dealloc(RecPropObject *self) {
    Py_XDECREF(self->fxn);
    Py_XDECREF(self->nm);
    Py_XDECREF(self->doc);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int recprop_init(RecPropObject *self, PyObject *args, PyObject *kwds) {
    PyObject *fxn;
    if (!PyArg_ParseTuple(args, "O", &fxn)) return -1;
    if (init_strs() < 0) return -1;
    Py_INCREF(fxn);
    self->fxn = fxn;
    /* nm = fxn.__name__ */
    self->nm = PyObject_GetAttrString(fxn, "__name__");
    if (!self->nm) return -1;
    PyUnicode_InternInPlace(&self->nm);
    /* doc = fxn.__doc__ */
    self->doc = PyObject_GetAttrString(fxn, "__doc__");
    if (!self->doc) { PyErr_Clear(); self->doc = Py_None; Py_INCREF(Py_None); }
    return 0;
}

/* __set_name__(self, owner, name) */
static PyObject* recprop_set_name(RecPropObject *self, PyObject *args) {
    PyObject *owner, *name;
    if (!PyArg_ParseTuple(args, "OO", &owner, &name)) return NULL;
    Py_DECREF(self->nm);
    Py_INCREF(name);
    self->nm = name;
    PyUnicode_InternInPlace(&self->nm);
    Py_RETURN_NONE;
}

/* The core __get__ descriptor method — called on every attribute access that misses __dict__ */
static PyObject* recprop_descr_get(RecPropObject *self, PyObject *obj, PyObject *type) {
    if (obj == NULL || obj == Py_None) {
        Py_INCREF((PyObject *)self);
        return (PyObject *)self;
    }

    PyObject *nm = self->nm;

    /* Medium path: if all children already have this property, compute directly */
    PyObject *src = PyObject_GetAttr(obj, str_src);
    if (!src) return NULL;
    Py_ssize_t slen = PyTuple_GET_SIZE(src);

    int all_cached = 1;
    for (Py_ssize_t i = 0; i < slen; i++) {
        PyObject *child = PyTuple_GET_ITEM(src, i);
        PyObject *child_dict = PyObject_GenericGetDict(child, NULL);
        if (!child_dict) { Py_DECREF(src); return NULL; }
        int has = PyDict_Contains(child_dict, nm);
        Py_DECREF(child_dict);
        if (has < 0) { Py_DECREF(src); return NULL; }
        if (!has) { all_cached = 0; break; }
    }
    Py_DECREF(src);

    if (all_cached) {
        /* All children have the value — compute for this node and cache */
        PyObject *val = PyObject_CallOneArg(self->fxn, obj);
        if (!val) return NULL;
        PyObject *obj_dict = PyObject_GenericGetDict(obj, NULL);
        if (!obj_dict) { Py_DECREF(val); return NULL; }
        if (PyDict_SetItem(obj_dict, nm, val) < 0) { Py_DECREF(obj_dict); Py_DECREF(val); return NULL; }
        Py_DECREF(obj_dict);
        return val;  /* new ref from CallOneArg */
    }

    /* Slow path: iterative DFS computing property bottom-up (avoids tuple creation) */
    PyObject *fxn = self->fxn;

    /* Stack: parallel arrays */
    Py_ssize_t cap = 1024, size = 0;
    PyObject **nodes = (PyObject **)PyMem_Malloc(cap * sizeof(PyObject *));
    int *visited = (int *)PyMem_Malloc(cap * sizeof(int));
    if (!nodes || !visited) { PyMem_Free(nodes); PyMem_Free(visited); return PyErr_NoMemory(); }

    #define RP_PUSH(n, v) do { \
        if (size >= cap) { \
            cap *= 2; \
            PyObject **tn = (PyObject **)PyMem_Realloc(nodes, cap * sizeof(PyObject *)); \
            int *tv = (int *)PyMem_Realloc(visited, cap * sizeof(int)); \
            if (!tn || !tv) { PyMem_Free(nodes); PyMem_Free(visited); return PyErr_NoMemory(); } \
            nodes = tn; visited = tv; \
        } \
        nodes[size] = (n); visited[size] = (v); size++; \
    } while(0)

    RP_PUSH(obj, 0);

    while (size > 0) {
        size--;
        PyObject *node = nodes[size];
        int vis = visited[size];

        /* check if already in __dict__ */
        PyObject *node_dict = PyObject_GenericGetDict(node, NULL);
        if (!node_dict) goto slow_error;
        int has = PyDict_Contains(node_dict, nm);
        Py_DECREF(node_dict);
        if (has < 0) goto slow_error;
        if (has) continue;

        if (!vis) {
            RP_PUSH(node, 1);
            PyObject *nsrc = PyObject_GetAttr(node, str_src);
            if (!nsrc) goto slow_error;
            Py_ssize_t nslen = PyTuple_GET_SIZE(nsrc);
            for (Py_ssize_t i = nslen - 1; i >= 0; i--) {
                PyObject *child = PyTuple_GET_ITEM(nsrc, i);
                PyObject *cd = PyObject_GenericGetDict(child, NULL);
                if (!cd) { Py_DECREF(nsrc); goto slow_error; }
                int ch = PyDict_Contains(cd, nm);
                Py_DECREF(cd);
                if (ch < 0) { Py_DECREF(nsrc); goto slow_error; }
                if (!ch) { RP_PUSH(child, 0); }
            }
            Py_DECREF(nsrc);
        } else {
            /* compute and cache */
            PyObject *val = PyObject_CallOneArg(fxn, node);
            if (!val) goto slow_error;
            PyObject *nd = PyObject_GenericGetDict(node, NULL);
            if (!nd) { Py_DECREF(val); goto slow_error; }
            if (PyDict_SetItem(nd, nm, val) < 0) { Py_DECREF(nd); Py_DECREF(val); goto slow_error; }
            Py_DECREF(nd);
            Py_DECREF(val);
        }
    }

    #undef RP_PUSH
    PyMem_Free(nodes);
    PyMem_Free(visited);

    /* Return cached value from obj.__dict__ */
    {
        PyObject *obj_dict = PyObject_GenericGetDict(obj, NULL);
        if (!obj_dict) return NULL;
        PyObject *ret = PyDict_GetItemWithError(obj_dict, nm);
        Py_DECREF(obj_dict);
        if (!ret) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "recursive_property: value not in __dict__ after DFS");
            return NULL;
        }
        Py_INCREF(ret);
        return ret;
    }

slow_error:
    PyMem_Free(nodes);
    PyMem_Free(visited);
    return NULL;
}

static PyMemberDef recprop_members[] = {
    {"fxn", T_OBJECT_EX, offsetof(RecPropObject, fxn), READONLY, "wrapped function"},
    {"nm", T_OBJECT_EX, offsetof(RecPropObject, nm), READONLY, "property name"},
    {"__doc__", T_OBJECT_EX, offsetof(RecPropObject, doc), 0, "docstring"},
    {NULL}
};

static PyMethodDef recprop_methods[] = {
    {"__set_name__", (PyCFunction)recprop_set_name, METH_VARARGS, "set property name"},
    {NULL}
};

static PyTypeObject RecPropType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "tinygrad.uop._fast_recursive_property.recursive_property",
    .tp_basicsize = sizeof(RecPropObject),
    .tp_dealloc = (destructor)recprop_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "C recursive_property descriptor",
    .tp_methods = recprop_methods,
    .tp_members = recprop_members,
    .tp_init = (initproc)recprop_init,
    .tp_new = PyType_GenericNew,
    .tp_descr_get = (descrgetfunc)recprop_descr_get,
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_fast_recursive_property", NULL, -1, NULL
};

PyMODINIT_FUNC PyInit__fast_recursive_property(void) {
    if (PyType_Ready(&RecPropType) < 0) return NULL;
    PyObject *m = PyModule_Create(&module);
    if (!m) return NULL;
    Py_INCREF(&RecPropType);
    if (PyModule_AddObject(m, "recursive_property", (PyObject *)&RecPropType) < 0) {
        Py_DECREF(&RecPropType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
