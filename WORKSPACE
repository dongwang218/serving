local_repository(
  name = "tf",
  path = __workspace_dir__ + "/tensorflow",
)

load('//tensorflow/tensorflow:workspace.bzl', 'tf_workspace')
tf_workspace("tensorflow/")

# ===== gRPC dependencies =====

bind(
    name = "libssl",
    actual = "@boringssl_git//:ssl",
)

git_repository(
    name = "boringssl_git",
    commit = "436432d849b83ab90f18773e4ae1c7a8f148f48d",
    init_submodules = True,
    remote = "https://github.com/mdsteele/boringssl-bazel.git",
)

bind(
    name = "protobuf_clib",
    actual = "@tf//google/protobuf:protobuf",
)

bind(
    name = "zlib",
    actual = "@zlib_archive//:zlib",
)

new_http_archive(
    name = "zlib_archive",
    build_file = "zlib.BUILD",
    sha256 = "879d73d8cd4d155f31c1f04838ecd567d34bebda780156f0e82a20721b3973d5",
    strip_prefix = "zlib-1.2.8",
    url = "http://zlib.net/zlib128.zip",
)
