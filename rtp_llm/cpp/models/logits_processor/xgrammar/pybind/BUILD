load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_library(
    name = "xgrammar_bootstrap",
    srcs = ["XGrammarBootstrap.cc"],
    hdrs = ["XGrammarBootstrap.h"],
    deps = [
        "@xgrammar//:xgrammar",
    ] + torch_deps(),
    copts = copts(),
    alwayslink = True,
    visibility = ["//visibility:public"],
)
