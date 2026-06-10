load("//:def.bzl", "cc_test_wrapper", "copts")

cc_test = cc_test_wrapper

cc_test(
    name = "structured_output_admission_policy_test",
    srcs = ["StructuredOutputAdmissionPolicyTest.cc"],
    copts = [
        "-fno-access-control",
    ] + copts(),
    deps = [
        "//rtp_llm/cpp/engine_base:structured_output_admission",
        "@com_google_googletest//:gtest_main",
    ],
)
