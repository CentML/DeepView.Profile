'use babel';

export default class EventType {
  constructor({category, action}) {
    this._category = category;
    this._action = action;
  }

  static of(category, action) {
    return new EventType({category, action});
  }

  get category() {
    return this._category;
  }

  get action() {
    return this._action;
  }

  get name() {
    return `${this.category} / ${this.action}`;
  }
};
