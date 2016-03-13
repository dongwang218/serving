# Description: Tensorflow Serving examples.

package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = [
        "-parse_headers",
        "no_layering_check",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

cc_library(
    name = "inception_grpc_cc",
    srcs = [
        "inception_inference.grpc.pb.cc",
        "inception_inference.pb.cc",
    ],
    hdrs = [
        "inception_inference.grpc.pb.h",
        "inception_inference.pb.h",
    ],
    deps = [
        "//grpc:grpc++",
    ],
)

py_binary(
    name = "inception_export",
    srcs = [
        "inception_export.py",
    ],
    deps = [
        "@tf//tensorflow:tensorflow_py",
        "//tensorflow_serving/session_bundle:exporter",
    ],
)

cc_binary(
    name = "inception_inference",
    srcs = [
        "inception_inference.cc",
    ],
    linkopts = ["-lm"],
    deps = [
        "@tf//tensorflow/core:core_cpu",
        "@tf//tensorflow/core:framework",
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:protos_all_cc",
        "@tf//tensorflow/core:tensorflow",
        ":inception_grpc_cc",
        "//grpc",
        "//tensorflow_serving/session_bundle",
        "//tensorflow_serving/session_bundle:manifest_proto",
        "//tensorflow_serving/session_bundle:signature",
    ],
)

py_binary(
    name = "inception_client",
    srcs = [
        "inception_client.py",
        "inception_inference_pb2.py",
    ],
    deps = [
        "@tf//tensorflow:tensorflow_py"
    ],
)
