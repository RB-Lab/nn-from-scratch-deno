import { Driver, Matrix, Vector } from "./driver.ts";
import { } from "https://deno.land/x/neo@0.0.1-pre.1/mod.ts";

export const simpleDriver: Driver = {
  vector: function (size: number | number[]): Vector {
    throw new Error("Function not implemented.");
  },
  matrix: function (height: number | number[][], width?: number): Matrix {
    throw new Error("Function not implemented.");
  },
  matmul: function (a: Matrix, b: Matrix): Matrix {
    throw new Error("Function not implemented.");
  },
  transpose: function (a: Matrix): Matrix {
    throw new Error("Function not implemented.");
  },
  addVector: function (a: Matrix, b: Vector): Matrix {
    throw new Error("Function not implemented.");
  },
  sumColumns: function (a: Matrix): Vector {
    throw new Error("Function not implemented.");
  },
  copy: function (a: Matrix): Matrix {
    throw new Error("Function not implemented.");
  },
  oneHot: function (y: Vector, classes: number): Matrix {
    throw new Error("Function not implemented.");
  },
  activation: {
    relu: {
      forward: function (x: Matrix): Matrix {
        throw new Error("Function not implemented.");
      },
      backward: function (x: Matrix): Matrix {
        throw new Error("Function not implemented.");
      },
    },
    softmax: {
      forward: function (x: Matrix): Matrix {
        throw new Error("Function not implemented.");
      },
      backward: function (x: Matrix): Matrix {
        throw new Error("Function not implemented.");
      },
    },
  },
  loss: {
    crossEntropy: {
      forward: function (yTrue: Vector, yPredBatch: Matrix): Vector {
        throw new Error("Function not implemented.");
      },
      backward: function (yTrue: Vector, yPredBatch: Matrix): Matrix {
        throw new Error("Function not implemented.");
      },
    },
  },
  combo: {
    softMaxCrossEntropy: {
      forward: function (
        yTrue: Vector,
        yPredBatch: Matrix
      ): { losses: Vector; yPred: Matrix } {
        throw new Error("Function not implemented.");
      },
      backward: function (yTrue: Vector, yPredBatch: Matrix): Matrix {
        throw new Error("Function not implemented.");
      },
    },
  },
  math: {
    mean: function (values: Vector): number {
      throw new Error("Function not implemented.");
    },
  },
};
