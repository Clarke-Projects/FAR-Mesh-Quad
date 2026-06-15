
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static wchar_t *decode_arg(const char *arg) {
    wchar_t *out = Py_DecodeLocale(arg, NULL);
    if (!out) {
        fprintf(stderr, "Failed to decode argument: %s\n", arg ? arg : "(null)");
        exit(120);
    }
    return out;
}

static void prepend_env_path(const char *name, const char *prefix) {
    const char *old_value = getenv(name);
    size_t need = strlen(prefix) + 2 + (old_value ? strlen(old_value) : 0);
    char *value = (char *)calloc(need, 1);
    if (!value) {
        fprintf(stderr, "Out of memory\n");
        exit(121);
    }

    if (old_value && old_value[0]) {
        snprintf(value, need, "%s:%s", prefix, old_value);
    } else {
        snprintf(value, need, "%s", prefix);
    }

    setenv(name, value, 1);
    free(value);
}

int main(int argc, char **argv) {
    const char *app_root = getenv("FAR_MESH_NATIVE_APP_ROOT");
    if (!app_root || !app_root[0]) {
        app_root = "/opt/far-mesh-quad-native";
    }

    char quadwild_root[4096];
    char quadwild_lib[4096];

    snprintf(quadwild_root, sizeof(quadwild_root), "%s/quadwild-bimdf", app_root);
    snprintf(quadwild_lib, sizeof(quadwild_lib), "%s/quadwild-bimdf/build/Build/lib", app_root);

    setenv("FAR_MESH_APP_ROOT", app_root, 1);
    setenv("FAR_MESH_QUADWILD_ROOT", quadwild_root, 1);

    unsetenv("VTK_DEFAULT_OPENGL_WINDOW");
    unsetenv("VTK_USE_EGL");
    unsetenv("VTK_RENDER_WINDOW_TYPE");

    if (!getenv("QT_QPA_PLATFORM")) {
        setenv("QT_QPA_PLATFORM", "xcb", 1);
    }

    prepend_env_path("LD_LIBRARY_PATH", quadwild_lib);

    /*
      Important:
      Keep user-site disabled for reproducibility.
      Multiprocessing helpers are forced to /usr/bin/python below,
      so the wrapper itself is not recursively launched as a worker.
    */
    setenv("PYTHONNOUSERSITE", "1", 0);

    if (chdir(app_root) != 0) {
        perror("chdir app_root failed");
        return 122;
    }

    const char *bootstrap =
        "import multiprocessing as _mp\n"
        "_mp.set_executable(\"/usr/bin/python\")\n"
        "import runpy\n"
        "runpy.run_module(\"far_mesh.main\", run_name=\"__main__\", alter_sys=True)\n";

    int py_argc = argc + 2;
    wchar_t **py_argv = (wchar_t **)calloc((size_t)py_argc + 1, sizeof(wchar_t *));
    if (!py_argv) {
        fprintf(stderr, "Out of memory\n");
        return 123;
    }

    py_argv[0] = decode_arg("far-mesh-quad-native");
    py_argv[1] = decode_arg("-c");
    py_argv[2] = decode_arg(bootstrap);

    for (int i = 1; i < argc; ++i) {
        py_argv[i + 2] = decode_arg(argv[i]);
    }

    int rc = Py_Main(py_argc, py_argv);

    for (int i = 0; i < py_argc; ++i) {
        PyMem_RawFree(py_argv[i]);
    }
    free(py_argv);

    return rc;
}
