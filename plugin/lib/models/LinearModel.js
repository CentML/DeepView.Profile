'use babel';

export default class LinearModel {
  constructor({slope, bias}) {
    this._slope = slope;
    this._bias = bias;
  }

  get slope() {
    return this._slope;
  }

  get bias() {
    return this._bias;
  }

  evaluate(x) {
    return this._slope * x + this._bias;
  }

  static fromProtobuf(protobufLinearModel) {
    if (protobufLinearModel == null) {
      return null;
    }
    return new LinearModel({
      slope: protobufLinearModel.getSlope(),
      bias: protobufLinearModel.getBias(),
    });
  }
};
