#pragma once

#include "../../layers/upsample.hh"
#include "../../tensor.hh"
#include "bottleneck.hh"
#include "c3k.hh"
#include "c3k2.hh"
#include "cbs.hh"
#include "concat.hh"
#include "sppf.hh"
#include <vector>

namespace vhn::cpu::yolov11 {

template <typename T> class YOLOv11n {
public:
  YOLOv11n()
      // Backbone
      : _stem(3, 16, 3, 2, 1),      // 640 -> 320
        _conv1(16, 32, 3, 2, 1),    // 320 -> 160
        _c3k2_1(32, 64, 1),         // 160
        _conv2(64, 128, 3, 2, 1),   // 160 -> 80
        _c3k2_2(128, 128, 1),       // 80
        _conv3(128, 256, 3, 2, 1),  // 80 -> 40
        _c3k2_3(256, 256, 1),       // 40
        _conv4(256, 512, 3, 2, 1),  // 40 -> 20
        _c3k2_4(512, 512, 1),       // 20
        _conv5(512, 1024, 3, 2, 1), // 20 -> 10
        _sppf(1024, 512),           // 10 (SPPF)
        _c2psa(512, 512),           // 10

        // Neck
        _upsample1(2),                     // 10 -> 20
        _concat1(1), _c3k2_5(768, 256, 1), // 20

        _upsample2(2),                     // 20 -> 40
        _concat2(1), _c3k2_6(384, 128, 1), // 40

        _conv6(128, 128, 3, 2, 1),         // 40 -> 20
        _concat3(1), _c3k2_7(384, 256, 1), // 20

        _conv7(256, 256, 3, 2, 1),           // 20 -> 10
        _concat4(1), _c3k2_8(768, 512, 1) {} // 10

  void load_weights(const std::vector<std::vector<std::vector<T>>> &weights) {}

  Tensor<T> forward(const Tensor<T> &input) {
    // Backbone
    auto x0 = _stem.forward(input); // [3, 640, 640] -> [16, 320, 320]
    auto x1 = _conv1.forward(x0);   // [16, 320, 320] -> [32, 160, 160]
    auto x2 = _c3k2_1.forward(x1);  // [32, 160, 160] -> [64, 160, 160]
    auto x3 = _conv2.forward(x2);   // [64, 160, 160] -> [128, 80, 80]
    auto x4 = _c3k2_2.forward(x3);  // [128, 80, 80] -> [128, 80, 80]
    auto x5 = _conv3.forward(x4);   // [128, 80, 80] -> [256, 40, 40]
    auto x6 = _c3k2_3.forward(x5);  // [256, 40, 40] -> [256, 40, 40]
    auto x7 = _conv4.forward(x6);   // [256, 40, 40] -> [512, 20, 20]
    auto x8 = _c3k2_4.forward(x7);  // [512, 20, 20] -> [512, 20, 20]
    auto x9 = _conv5.forward(x8);   // [512, 20, 20] -> [1024, 10, 10]

    // Neck
    auto sppf_out = _sppf.forward(x9); // [1024, 10, 10] -> [512, 10, 10]
    auto c2psa_out = _c2psa.forward(sppf_out); // [512, 10, 10] -> [512, 10, 10]

    auto up1 = _upsample1.forward(c2psa_out); // [512, 10, 10] -> [512, 20, 20]
    auto concat1 = _concat1.forward(
        {up1, x8}); // [512, 20, 20] + [512, 20, 20] -> [1024, 20, 20]
    auto p3 = _c3k2_5.forward(concat1); // [1024, 20, 20] -> [256, 20, 20]

    auto up2 = _upsample2.forward(p3); // [256, 20, 20] -> [256, 40, 40]
    auto concat2 = _concat2.forward(
        {up2, x6}); // [256, 40, 40] + [256, 40, 40] -> [512, 40, 40]
    auto p4 = _c3k2_6.forward(concat2); // [512, 40, 40] -> [128, 40, 40]

    auto down1 = _conv6.forward(p4); // [128, 40, 40] -> [128, 20, 20]
    auto concat3 = _concat3.forward(
        {down1, p3}); // [128, 20, 20] + [256, 20, 20] -> [384, 20, 20]
    auto p5 = _c3k2_7.forward(concat3); // [384, 20, 20] -> [256, 20, 20]

    auto down2 = _conv7.forward(p5); // [256, 20, 20] -> [256, 10, 10]
    auto concat4 = _concat4.forward(
        {down2, c2psa_out}); // [256, 10, 10] + [512, 10, 10] -> [768, 10, 10]
    auto p6 = _c3k2_8.forward(concat4); // [768, 10, 10] -> [512, 10, 10]

    return p6;
  }

  Tensor<T> forward_batched(const Tensor<T> &input) {
    // Backbone
    auto x0 =
        _stem.forward_batched(input); // [N, 3, 640, 640] -> [N, 16, 320, 320]
    auto x1 =
        _conv1.forward_batched(x0); // [N, 16, 320, 320] -> [N, 32, 160, 160]
    auto x2 =
        _c3k2_1.forward_batched(x1); // [N, 32, 160, 160] -> [N, 64, 160, 160]
    auto x3 =
        _conv2.forward_batched(x2); // [N, 64, 160, 160] -> [N, 128, 80, 80]
    auto x4 =
        _c3k2_2.forward_batched(x3); // [N, 128, 80, 80] -> [N, 128, 80, 80]
    auto x5 =
        _conv3.forward_batched(x4); // [N, 128, 80, 80] -> [N, 256, 40, 40]
    auto x6 =
        _c3k2_3.forward_batched(x5); // [N, 256, 40, 40] -> [N, 256, 40, 40]
    auto x7 =
        _conv4.forward_batched(x6); // [N, 256, 40, 40] -> [N, 512, 20, 20]
    auto x8 =
        _c3k2_4.forward_batched(x7); // [N, 512, 20, 20] -> [N, 512, 20, 20]
    auto x9 =
        _conv5.forward_batched(x8); // [N, 512, 20, 20] -> [N, 1024, 10, 10]

    // Neck
    auto sppf_out =
        _sppf.forward_batched(x9); // [N, 1024, 10, 10] -> [N, 512, 10, 10]
    auto c2psa_out = _c2psa.forward_batched(
        sppf_out); // [N, 512, 10, 10] -> [N, 512, 10, 10]

    auto up1 = _upsample1.forward_batched(
        c2psa_out); // [N, 512, 10, 10] -> [N, 512, 20, 20]
    auto concat1 = _concat1.forward(
        {up1, x8}); // [N, 512, 20, 20] + [N, 512, 20, 20] -> [N, 1024, 20, 20]
    auto p3 = _c3k2_5.forward_batched(
        concat1); // [N, 1024, 20, 20] -> [N, 256, 20, 20]

    auto up2 =
        _upsample2.forward_batched(p3); // [N, 256, 20, 20] -> [N, 256, 40, 40]
    auto concat2 = _concat2.forward(
        {up2, x6}); // [N, 256, 40, 40] + [N, 256, 40, 40] -> [N, 512, 40, 40]
    auto p4 = _c3k2_6.forward_batched(
        concat2); // [N, 512, 40, 40] -> [N, 128, 40, 40]

    auto down1 =
        _conv6.forward_batched(p4); // [N, 128, 40, 40] -> [N, 128, 20, 20]
    auto concat3 = _concat3.forward(
        {down1, p3}); // [N, 128, 20, 20] + [N, 256, 20, 20] -> [N, 384, 20, 20]
    auto p5 = _c3k2_7.forward_batched(
        concat3); // [N, 384, 20, 20] -> [N, 256, 20, 20]

    auto down2 =
        _conv7.forward_batched(p5); // [N, 256, 20, 20] -> [N, 256, 10, 10]
    auto concat4 = _concat4.forward(
        {down2,
         c2psa_out}); // [N, 256, 10, 10] + [N, 512, 10, 10] -> [N, 768, 10, 10]
    auto p6 = _c3k2_8.forward_batched(
        concat4); // [N, 768, 10, 10] -> [N, 512, 10, 10]

    return p6;
  }

  [[nodiscard]] const CBS<T> &stem() const { return _stem; }
  [[nodiscard]] const SPPF<T> &sppf() const { return _sppf; }
  [[nodiscard]] const C3k2<T> &c3k2_1() const { return _c3k2_1; }
  [[nodiscard]] const C3k2<T> &c3k2_2() const { return _c3k2_2; }
  [[nodiscard]] const C3k2<T> &c3k2_3() const { return _c3k2_3; }
  [[nodiscard]] const C3k2<T> &c3k2_4() const { return _c3k2_4; }
  [[nodiscard]] const C3k2<T> &c3k2_5() const { return _c3k2_5; }
  [[nodiscard]] const C3k2<T> &c3k2_6() const { return _c3k2_6; }
  [[nodiscard]] const C3k2<T> &c3k2_7() const { return _c3k2_7; }
  [[nodiscard]] const C3k2<T> &c3k2_8() const { return _c3k2_8; }

private:
  // Backbone
  CBS<T> _stem;
  CBS<T> _conv1, _conv2, _conv3, _conv4, _conv5;
  C3k2<T> _c3k2_1, _c3k2_2, _c3k2_3, _c3k2_4;
  SPPF<T> _sppf;
  C3k2<T> _c2psa;

  // Neck
  Upsample<T> _upsample1, _upsample2;
  Concat<T> _concat1, _concat2, _concat3, _concat4;
  C3k2<T> _c3k2_5, _c3k2_6;

  CBS<T> _conv6, _conv7;
  C3k2<T> _c3k2_7, _c3k2_8;
};

} // namespace vhn::cpu::yolov11
