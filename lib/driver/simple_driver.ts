import { Driver, Matrix, Vector } from "./driver.ts";

export const simpleDriver: Driver = {
  loss: {
    crossEntropy: {
      forward: (y: Vector, yPredBatch: Matrix) => {
        return y.map((label, j) => {
          const predicted = yPredBatch[j];
          return -Math.log(predicted[label]);
        });
      },
      backward: (yTrue: Vector, yPredBatch: Matrix) => {
        return yTrue.map((label, j) => {
          const predicted = yPredBatch[j];
          const result = predicted.map((value, i) => {
            if (i === label) {
              return -1 / value / yTrue.length;
            }
            return 0;
          });
          return result;
        });
      },
    },
  },
  activation: {
    relu: {
      forward: (input: Matrix) => {
        return input.map((row) => row.map((value) => Math.max(value, 0)));
      },
      backward: (input: Matrix, grad: Matrix) => {
        return grad.map((row, i) => {
          return row.map((value, j) => (input[i][j] > 0 ? value : 0));
        });
      },
    },
    softmax: {
      forward: (input: Matrix) => {
        const max = input.map((row) => Math.max(...row));
        const exp = input.map((row, i) =>
          row.map((value) => Math.exp(value - max[i]))
        );
        const sum = exp.map((row) => row.reduce((a, b) => a + b));
        return exp.map((row, i) => row.map((value) => clip(value / sum[i])));
      },
      backward: (output: Matrix, grad: Matrix) => {
        return output.map((oneSoftmax, i) => {
          const oneGrad = grad[i];
          // Here goes Jacobian matrix – a matrix of all partial derivatives of the outputs
          // vector with respect to inputs: ∂Sᵢⱼ/∂Zᵢⱼ = Sᵢⱼ * (𝛅ᵢₖ - Sᵢₖ)
          const jacobianMatrix: Matrix = [];
          for (let j = 0; j < oneSoftmax.length; j++) {
            if (!jacobianMatrix[j]) jacobianMatrix[j] = [];
            for (let k = 0; k < oneSoftmax.length; k++) {
              const kd = j === k ? 1 : 0; // kronecker delta
              jacobianMatrix[j][k] = oneSoftmax[j] * (kd - oneSoftmax[k]);
            }
          }
          return jacobianMatrix.map((row) => {
            return row.reduce((sum, d, k) => sum + d * oneGrad[k], 0);
          });
        });
      },
    },
  },

  combo: {
    softMaxCrossEntropy: {
      forward: (yTrue: Vector, yPredBatch: Matrix) => {
        const yPred = simpleDriver.activation.softmax.forward(yPredBatch);
        const losses = simpleDriver.loss.crossEntropy.forward(yTrue, yPred);
        return { losses, yPred };
      },
      backward: (yTrue: Vector, yPredBatch: Matrix) => {
        // derivative of cross entropy and softmax with respect to softmax input
        // is softmax yPredᵢₖ - yTrueᵢₖ, where yTrueᵢₖ is one-hot encoded, it basically
        // means that yTrueᵢₖ is 1 if k is the correct class and 0 otherwise, but
        // we have yTrue as a vector of correct class indices, so we can just
        // for each sample in yPredBatch subtract 1 from predicted value at index yTrue points to
        return yPredBatch.map((sample, i) => {
          const newRow = sample.slice();
          newRow[yTrue[i]] -= 1;
          return newRow.map((value) => value / yTrue.length);
        });
      },
    },
  },
  vector(array: number | number[]): Vector {
    if (Array.isArray(array)) {
      return array;
    }
    if (typeof array === "number") {
      return new Array(array).fill(0);
    }
    throw new Error(
      `Wrong parameters type: expected number or number[] ` +
        `but ${typeof array} given`
    );
  },

  matrix(height: number | number[][], width?: number): Matrix {
    if (Array.isArray(height)) {
      return height;
    }
    if (typeof height === "number" && typeof width === "number") {
      const result: Matrix = [];
      for (let i = 0; i < height; i++) {
        result[i] = [];
        for (let j = 0; j < width; j++) {
          result[i][j] = Math.random();
        }
      }
      return result;
    }
    throw new Error(
      `Wrong parameters type: expected number, number or number[][] ` +
        `but ${typeof height}, ${typeof width} given`
    );
  },

  matmul(a: Matrix, b: Matrix): Matrix {
    const result: Matrix = [];
    for (let i = 0; i < a.length; i++) {
      result[i] = [];
      for (let j = 0; j < b[0].length; j++) {
        let sum = 0;
        for (let k = 0; k < a[0].length; k++) {
          sum += a[i][k] * b[k][j];
        }
        result[i][j] = sum;
      }
    }
    return result;
  },

  transpose(a: Matrix): Matrix {
    const result: Matrix = [];
    for (let i = 0; i < a.length; i++) {
      for (let j = 0; j < a[0].length; j++) {
        if (!result[j]) {
          result[j] = [];
        }
        result[j][i] = a[i][j];
      }
    }
    return result;
  },

  addVector(a: Matrix, b: Vector): Matrix {
    return a.map((row) => row.map((value, j) => value + b[j]));
  },
  sumColumns(a: Matrix): Vector {
    const result: Vector = [];
    for (let i = 0; i < a[0].length; i++) {
      let sum = 0;
      for (let j = 0; j < a.length; j++) {
        sum += a[j][i];
      }
      result[i] = sum;
    }
    return result;
  },
  copy(a: Matrix): Matrix {
    return a.map((row) => row.slice());
  },
  oneHot: function (y: Vector, classes: number): Matrix {
    return y.map((label) => {
      const row = new Array(classes).fill(0);
      row[label] = 1;
      return row;
    });
  },
  math: {
    mean: function (values: Vector): number {
      return values.reduce((a, b) => a + b) / values.length;
    },
  },
};

function clip(x: number, min = 1e-7, max = 1 - 1e-7) {
  if (x < min) return min;
  if (x > max) return max;
  return x;
}
