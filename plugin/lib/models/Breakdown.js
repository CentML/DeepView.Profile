'use babel';

import {processFileReference} from '../utils';

class BreakdownNode {
  constructor({id, name, contexts}) {
    this._id = id;
    this._name = name;
    this._contexts = contexts;
    this._parent = null;
    this._children = [];
  }

  get id() {
    return this._id;
  }

  get name() {
    return this._name;
  }

  get contexts() {
    return this._contexts;
  }

  get parent() {
    return this._parent;
  }

  get children() {
    return this._children;
  }
}

export class OperationNode extends BreakdownNode {
  constructor({id, name, contexts, forwardMs, backwardMs, sizeBytes}) {
    super({id, name, contexts});
    this._forwardMs = forwardMs;
    this._backwardMs = backwardMs;
    this._sizeBytes = sizeBytes;
  }

  get runTimeMs() {
    if (isNaN(this._backwardMs)) {
      return this._forwardMs;
    }
    return this._forwardMs + this._backwardMs;
  }

  get forwardMs() {
    return this._forwardMs;
  }

  get backwardMs() {
    return this._backwardMs;
  }

  get sizeBytes() {
    return this._sizeBytes;
  }

  static fromProtobufNodeList(operationNodeList) {
    return constructTree(
      operationNodeList,
      OperationNode._fromProtobufNode,
    );
  }

  static _fromProtobufNode(protobufBreakdownNode, id) {
    return new OperationNode({
      id,
      name: protobufBreakdownNode.getName(),
      contexts: protobufBreakdownNode.getContextsList().map(processFileReference),
      ...parseOperationData(protobufBreakdownNode),
    });
  }
};

export class WeightNode extends BreakdownNode {
  constructor({id, name, contexts, sizeBytes, gradSizeBytes}) {
    super({id, name, contexts});
    this._sizeBytes = sizeBytes;
    this._gradSizeBytes = gradSizeBytes;
  }

  get sizeBytes() {
    return this._sizeBytes;
  }

  get gradSizeBytes() {
    return this._gradSizeBytes;
  }

  static fromProtobufNodeList(weightNodeList) {
    return constructTree(
      weightNodeList,
      WeightNode._fromProtobuf,
    );
  }

  static _fromProtobuf(protobufBreakdownNode, id) {
    return new WeightNode({
      id,
      name: protobufBreakdownNode.getName(),
      contexts: protobufBreakdownNode.getContextsList().map(processFileReference),
      ...parseWeightData(protobufBreakdownNode),
    });
  }
};

function constructTree(protobufArray, constructNode) {
  if (protobufArray.length == 0) {
    throw new Error('Skyline received an empty breakdown tree! Please file a bug report.');
  }

  const helperRoot = {_children: []};
  const parentNodeStack = [helperRoot];
  const numChildrenStack = [1];

  // The nodes were placed into the array based on a preorder traversal of the tree
  protobufArray.forEach((protobufNode, index) => {
    const parent = parentNodeStack.pop();
    const numChildren = numChildrenStack.pop();

    // We use the serialization index as an identifier for the node
    const node = constructNode(protobufNode, index);
    node._parent = parent;
    parent._children.push(node);

    if (numChildren > 1) {
      parentNodeStack.push(parent);
      numChildrenStack.push(numChildren - 1);
    }

    if (protobufNode.getNumChildren() > 0 ) {
      parentNodeStack.push(node);
      numChildrenStack.push(protobufNode.getNumChildren());
    }
  });

  const root = helperRoot._children[0];
  root._parent = null;
  return root;
}

function parseOperationData(protobufNode) {
  const data = protobufNode.getOperation();
  return {
    forwardMs: data.getForwardMs(),
    backwardMs: data.getBackwardMs(),
    sizeBytes: data.getSizeBytes(),
  };
}

function parseWeightData(protobufNode) {
  const data = protobufNode.getWeight();
  return {
    sizeBytes: data.getSizeBytes(),
    gradSizeBytes: data.getGradSizeBytes(),
  };
}
