/* WebAssembly kernel for local additive feature contributions. */
(function (global) {
  "use strict";

  const moduleBytes = new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x07, 0x01, 0x60, 0x02, 0x7d, 0x7d, 0x01, 0x7d,
    0x03, 0x02, 0x01, 0x00,
    0x07, 0x0e, 0x01, 0x0a, 0x64, 0x69, 0x66, 0x66, 0x65,
    0x72, 0x65, 0x6e, 0x63, 0x65, 0x00, 0x00,
    0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01,
    0x93, 0x0b,
  ]);

  let kernelPromise;
  global.PolishMLShapWasm = {
    ready() {
      if (!kernelPromise) {
        kernelPromise = WebAssembly.instantiate(moduleBytes).then(
          ({ instance }) => instance.exports.difference
        );
      }
      return kernelPromise;
    },
    async contributions(features, baseline) {
      const difference = await this.ready();
      return features.map((value, index) => difference(value, baseline[index]));
    },
    async alignmentFlags(raw, corrected) {
      const difference = await this.ready();
      return raw.map((base, index) => (
        difference(base, corrected[index]) === 0 ? 0 : 1
      ));
    },
  };
})(window);
