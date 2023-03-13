import { createSpiral } from "https://raw.githubusercontent.com/RB-Lab/deno-simple-datasets/master/mod.ts";

const { X, y } = createSpiral({
  classes: 3,
  samples: 100,
  twist: 5,
  dispersion: 0.1,
});
console.log(JSON.stringify({ X, y }));
