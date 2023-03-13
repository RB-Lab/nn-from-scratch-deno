import {
  assertAlmostEquals,
  assertEquals,
} from "https://deno.land/std@0.163.0/testing/asserts.ts";
import { simpleDriver } from "./simple_driver.ts";

const drivers = [simpleDriver];

Deno.test("matrix", () => {
  drivers.forEach((driver) => {
    const matrix = driver.matrix(2, 3);
    assertEquals(matrix.length, 2);
    assertEquals(matrix[0].length, 3);
    assertEquals(matrix[1].length, 3);
  });
});

Deno.test("matmul", () => {
  drivers.forEach((driver) => {
    const matrixA = [
      [1, 2, 3],
      [4, 5, 6],
    ];
    const matrixB = [
      [7, 8],
      [9, 10],
      [11, 12],
    ];
    const expected = [
      [7 * 1 + 9 * 2 + 11 * 3, 8 * 1 + 10 * 2 + 12 * 3],
      [7 * 4 + 9 * 5 + 11 * 6, 8 * 4 + 10 * 5 + 12 * 6],
    ];
    const result = driver.matmul(matrixA, matrixB);
    assertEquals(result, expected);
  });
});

Deno.test("transpose", () => {
  drivers.forEach((driver) => {
    const matrix = [
      [1, 2, 3],
      [4, 5, 6],
    ];
    const expected = [
      [1, 4],
      [2, 5],
      [3, 6],
    ];
    const result = driver.transpose(matrix);

    assertEquals(result, expected);
  });
});

Deno.test("addVector", () => {
  drivers.forEach((driver) => {
    const matrix = [
      [1, 2, 3],
      [4, 5, 6],
    ];
    const vector = [7, 8, 9];
    const expected = [
      [1 + 7, 2 + 8, 3 + 9],
      [4 + 7, 5 + 8, 6 + 9],
    ];
    const result = driver.addVector(matrix, vector);
    assertEquals(result, expected);
  });
});

Deno.test("activation.relu.forward", () => {
  drivers.forEach((driver) => {
    const x = [
      [-1, 2, -3],
      [4, -5, 6],
    ];
    const expected = [
      [0, 2, 0],
      [4, 0, 6],
    ];
    const result = driver.activation.relu.forward(x);
    assertEquals(result, expected);
  });
});

Deno.test("activation.relu.backward", () => {
  drivers.forEach((driver) => {
    const x = [
      [-1, 2, -3],
      [4, -5, 6],
    ];
    const grad = [
      [1, 2, 3],
      [-1, 5, 2],
    ];
    const expected = [
      [0, 2, 0],
      [-1, 0, 2],
    ];
    const result = driver.activation.relu.backward(x, grad);
    assertEquals(result, expected);
  });
});

Deno.test("activation.softmax.forward", () => {
  drivers.forEach((driver) => {
    const matrix = [
      [1, 2, 3],
      [4, 5, 6],
    ];
    const exps1 = [Math.exp(1), Math.exp(2), Math.exp(3)];
    const exps2 = [Math.exp(4), Math.exp(5), Math.exp(6)];
    const sum1 = exps1.reduce((a, b) => a + b);
    const sum2 = exps2.reduce((a, b) => a + b);
    const expected = [
      [exps1[0] / sum1, exps1[1] / sum1, exps1[2] / sum1],
      [exps2[0] / sum2, exps2[1] / sum2, exps2[2] / sum2],
    ];
    const result = driver.activation.softmax.forward(matrix);
    result.forEach((row, i) => {
      row.forEach((value, j) => {
        // we use assertAlmostEquals because exponentiation magnifies floating point glitches
        // default tolerance is 1e-7
        assertAlmostEquals(value, expected[i][j]);
      });
    });
  });
});

Deno.test("activation.softmax.backward", () => {
  drivers.forEach((driver) => {
    const output = [
      [0.3, 2, 3],
      [4, 5, 6],
    ];
    const grad = [
      [1, -2, 3],
      [-4, 5, -6],
    ];
    // Jacobian Matrix; derivative: Sᵢⱼ𝛅ᵢₖ - SᵢⱼSᵢₖ
    const jam = [
      [
        [
          // i = 0, k = 0
          output[0][0] * 1 - output[0][0] * output[0][0], // j = 0
          output[0][1] * 0 - output[0][1] * output[0][0], // j = 1
          output[0][2] * 0 - output[0][2] * output[0][0], // j = 2
        ],
        [
          // i = 0, k = 1
          output[0][0] * 0 - output[0][0] * output[0][1], // j = 0
          output[0][1] * 1 - output[0][1] * output[0][1], // j = 1
          output[0][2] * 0 - output[0][2] * output[0][1], // j = 2
        ],
        [
          // i = 0, k = 2
          output[0][0] * 0 - output[0][0] * output[0][2], // j = 0
          output[0][1] * 0 - output[0][1] * output[0][2], // j = 1
          output[0][2] * 1 - output[0][2] * output[0][2], // j = 2
        ],
      ],
      [
        [
          // i = 1, k = 0
          output[1][0] * 1 - output[1][0] * output[1][0], // j = 0
          output[1][1] * 0 - output[1][1] * output[1][0], // j = 1
          output[1][2] * 0 - output[1][2] * output[1][0], // j = 2
        ],
        [
          // i = 1, k = 1
          output[1][0] * 0 - output[1][0] * output[1][1], // j = 0
          output[1][1] * 1 - output[1][1] * output[1][1], // j = 1
          output[1][2] * 0 - output[1][2] * output[1][1], // j = 2
        ],
        [
          // i = 1, k = 2
          output[1][0] * 0 - output[1][0] * output[1][2], // j = 0
          output[1][1] * 0 - output[1][1] * output[1][2], // j = 1
          output[1][2] * 1 - output[1][2] * output[1][2], // j = 2
        ],
      ],
    ];
    // deno-fmt-ignore
    const expected = [
      [
        jam[0][0][0] * grad[0][0] + jam[0][0][1] * grad[0][1] + jam[0][0][2] * grad[0][2],
        jam[0][1][0] * grad[0][0] + jam[0][1][1] * grad[0][1] + jam[0][1][2] * grad[0][2],
        jam[0][2][0] * grad[0][0] + jam[0][2][1] * grad[0][1] + jam[0][2][2] * grad[0][2],
      ],
      [
        jam[1][0][0] * grad[1][0] + jam[1][0][1] * grad[1][1] + jam[1][0][2] * grad[1][2],
        jam[1][1][0] * grad[1][0] + jam[1][1][1] * grad[1][1] + jam[1][1][2] * grad[1][2],
        jam[1][2][0] * grad[1][0] + jam[1][2][1] * grad[1][1] + jam[1][2][2] * grad[1][2],
      ],
    ];
    const result = driver.activation.softmax.backward(output, grad);
    assertEquals(result, expected);
  });
});

Deno.test("loss.crossEntropy.forward", () => {
  drivers.forEach((driver) => {
    const matrix = [
      [0.1, 0.2, 0.7],
      [0.9, 0.05, 0.05],
    ];
    const labels = [2, 0];
    const expected = [-Math.log(matrix[0][2]), -Math.log(matrix[1][0])];
    const result = driver.loss.crossEntropy.forward(labels, matrix);
    assertEquals(result, expected);
  });
});

Deno.test("loss.crossEntropy.backward", () => {
  drivers.forEach((driver) => {
    const matrix = [
      [0.1, 0.2, 0.7],
      [0.9, 0.05, 0.05],
    ];
    const labels = [2, 0];
    const expected = [
      [0, 0, -1 / matrix[0][2] / matrix.length],
      [-1 / matrix[1][0] / matrix.length, 0, 0],
    ];
    const result = driver.loss.crossEntropy.backward(labels, matrix);
    assertEquals(result, expected);
  });
});

Deno.test("combo.softMaxCrossEntropy.forward", () => {
  drivers.forEach((driver) => {
    const matrix = [
      [0.1, 0.2, 0.7],
      [0.9, 0.05, 0.05],
    ];
    const labels = [2, 0];
    const expectedYPred = driver.activation.softmax.forward(matrix)
    const expectedLoss = driver.loss.crossEntropy.forward(labels, expectedYPred);
    const {losses, yPred} = driver.combo.softMaxCrossEntropy.forward(labels, matrix);
    assertEquals(losses, expectedLoss);
    assertEquals(yPred, expectedYPred);
  });
});

Deno.test("combo.softMaxCrossEntropy.backward", () => {
  drivers.forEach((driver) => {
    const matrix = [
      [0.1, 0.2, 0.7],
      [0.9, 0.05, 0.05],
    ];
    const labels = [2, 0];
    const softmaxOut = driver.activation.softmax.forward(matrix)
    const dCrossEntropy = driver.loss.crossEntropy.backward(labels, softmaxOut);
    const dSoftmax = driver.activation.softmax.backward(softmaxOut, dCrossEntropy);
    const result = driver.combo.softMaxCrossEntropy.backward(labels, softmaxOut)
    assertEquals(dSoftmax, result);
  });
});

Deno.test("math.mean", () => {
  drivers.forEach((driver) => {
    const vector = [1, 2, 3, 4, 5, 6];
    const expected = (1 + 2 + 3 + 4 + 5 + 6) / 6;
    const result = driver.math.mean(vector);
    assertEquals(result, expected);
  });
});

Deno.test("sumColumns", () => {
  drivers.forEach((driver) => {
    const matrix = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ];
    const expected = [1 + 4 + 7, 2 + 5 + 8, 3 + 6 + 9];
    const result = driver.sumColumns(matrix);
    assertEquals(result, expected);
  });
});

Deno.test("oneHot", () => {
  drivers.forEach((driver) => {
    const labels = [1, 0, 2, 1];
    const expected = [
      [0, 1, 0],
      [1, 0, 0],
      [0, 0, 1],
      [0, 1, 0],
    ];
    const result = driver.oneHot(labels, 3);
    assertEquals(result, expected);
  });
});
