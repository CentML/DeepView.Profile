'use babel';

import path from 'path';
import process from 'process';

import {CompositeDisposable} from 'atom';

// The FileTracker keeps track of all TextEditors and their associated files.
// The purpose of the FileTracker is to let us retrieve the TextEditor(s)
// associated with a given file.

export default class FileTracker {
  constructor({
    projectRoot,
    onOpenFilesChange = () => {},
    onProjectFileSave = () => {},
    onProjectModifiedChange = () => {},
  }) {
    // We only care about files that fall under the project root
    this._projectRoot = projectRoot;

    // Event callbacks
    this._onOpenFilesChange = onOpenFilesChange;
    this._onProjectFileSave = onProjectFileSave;
    this._onProjectModifiedChange = onProjectModifiedChange;

    // A mapping of all observed (and not yet destroyed) TextEditors to a
    // disposable that is used to remove our event subscriptions from that
    // editor. The keys of this map are a set of all the TextEditors that we
    // know about.
    //
    // Map<TextEditor, Disposable>
    this._subscriptionsByEditor = new Map();

    // Editor Index: Keeps track of TextEditors that display project files.

    // Stores the inverse mapping of the previous map. This map is part of
    // the editor index.
    //
    // Map<TextEditor, String>
    this._filePathByEditor = new Map();

    // A mapping of a project file path to TextEditor(s) that are currently
    // displaying the file. This map is part of the editor index.
    //
    // Map<String, List<TextEditor>>
    this._editorsByFilePath = new Multimap();

    // Stores all project TextEditors that have buffers with unsaved changes,
    // keyed by their file path. This set is part of the editor index.
    //
    // Map<string, List<TextEditor>>
    this._modifiedEditorsByFilePath = new Multimap();

    this._onNewEditor = this._onNewEditor.bind(this);
    this._subs = atom.workspace.observeTextEditors(this._onNewEditor);
  }

  dispose() {
    this._subs.dispose();
    this._subscriptionsByEditor.forEach(disposable => disposable.dispose());

    // We set to null to indicate that this tracker cannot be reused
    this._subscriptionsByEditor.clear();
    this._subscriptionsByEditor = null;

    this._editorsByFilePath.clear();
    this._editorsByFilePath = null;

    this._filePathByEditor.clear();
    this._filePathByEditor = null;

    this._modifiedEditorsByFilePath.clear();
    this._modifiedEditorsByFilePath = null;
  }

  getTextEditorsFor(filePath) {
    if (!this._editorsByFilePath.has(filePath)) {
      return [];
    }
    return this._editorsByFilePath.get(filePath);
  }

  hasModifiedFiles() {
    return this._modifiedEditorsByFilePath.size > 0;
  }

  modifiedEditorsByFilePath() {
    return this._modifiedEditorsByFilePath.copy();
  }

  editorsByFilePath() {
    return this._editorsByFilePath.copy();
  }

  editors() {
    return new Set(this._filePathByEditor.keys());
  }

  _onNewEditor(editor) {
    // Subscribe to relevant events on the TextEditor
    const subs = new CompositeDisposable();
    subs.add(editor.onDidChangePath(this._onEditorPathChange.bind(this, editor)));
    subs.add(editor.onDidSave(this._onEditorSave.bind(this, editor)));
    subs.add(editor.onDidChangeModified(this._onEditorModifiedChange.bind(this, editor)));
    subs.add(editor.onDidDestroy(this._onEditorDistroy.bind(this, editor)));
    this._subscriptionsByEditor.set(editor, subs);

    const callbacks = this._addEditorToIndexIfNeeded(editor);
    callbacks.forEach(callback => process.nextTick(callback));
  }

  _onEditorPathChange(editor) {
    const removalCallbacks = this._removeEditorFromIndexIfNeeded(editor);
    const additionCallbacks = this._addEditorToIndexIfNeeded(editor);
    const uniqueCallbacks = new Set([...removalCallbacks, ...additionCallbacks]);
    uniqueCallbacks.forEach(callback => process.nextTick(callback));
  }

  _onEditorSave(editor) {
    if (!this._filePathByEditor.has(editor)) {
      return;
    }
    process.nextTick(this._onProjectFileSave, editor);
  }

  _onEditorDistroy(editor) {
    const callbacks = this._removeEditorFromIndexIfNeeded(editor);
    this._subscriptionsByEditor.get(editor).dispose();
    this._subscriptionsByEditor.delete(editor);

    callbacks.forEach(callback => process.nextTick(callback));
  }

  _onEditorModifiedChange(editor) {
    const filePath = this._filePathByEditor.get(editor);
    if (filePath == null) {
      return;
    }

    if (editor.isModified()) {
      this._modifiedEditorsByFilePath.set(filePath, editor);
    } else {
      this._modifiedEditorsByFilePath.delete(filePath, editor);
    }

    process.nextTick(this._onProjectModifiedChange);
  }

  _toProjectRelativePath(candidateFilePath) {
    if (candidateFilePath == null || !candidateFilePath.startsWith(this._projectRoot)) {
      return null;
    }
    return path.relative(this._projectRoot, candidateFilePath);
  }

  _addEditorToIndexIfNeeded(editor) {
    // Abort if the editor is already in our index
    if (this._filePathByEditor.has(editor)) {
      return [];
    }

    // Abort if the editor is not displaying a project file
    const editorPathAbsolute = editor.getPath();
    const projectFilePath = this._toProjectRelativePath(editorPathAbsolute);
    if (projectFilePath == null) {
      return [];
    }

    // Update the editor index
    this._editorsByFilePath.set(projectFilePath, editor);
    this._filePathByEditor.set(editor, projectFilePath);

    const callbacks = [this._onOpenFilesChange];
    if (!editor.isModified()) {
      return callbacks;
    }

    // If the editor has unsaved changes, keep track of it
    this._modifiedEditorsByFilePath.set(projectFilePath, editor);
    callbacks.push(this._onProjectModifiedChange);

    return callbacks;
  }

  _removeEditorFromIndexIfNeeded(editor) {
    // Abort if the editor is not in our index
    if (!this._filePathByEditor.has(editor)) {
      return [];
    }

    const filePath = this._filePathByEditor.get(editor);
    this._filePathByEditor.delete(editor);
    this._editorsByFilePath.delete(filePath, editor);

    const callbacks = [this._onOpenFilesChange];
    if (!this._modifiedEditorsByFilePath.has(filePath)) {
      return callbacks;
    }

    this._modifiedEditorsByFilePath.delete(filePath, editor);
    callbacks.push(this._onProjectModifiedChange);
    return callbacks;
  }
}

class Multimap {
  constructor() {
    // Map<Key, List<Value>>
    this._map = new Map();
  }

  set(key, value) {
    const valueArray = this._map.get(key);
    if (valueArray == null) {
      this._map.set(key, [value]);
    } else {
      valueArray.push(value);
    }
  }

  delete(key, valueToDelete) {
    const valueArray = this._map.get(key);
    if (valueArray == null) {
      return;
    }

    const newValueArray = valueArray.filter(value => value !== valueToDelete);
    if (newValueArray.length == 0) {
      this._map.delete(key);
    } else {
      this._map.set(key, newValueArray);
    }
  }

  get size() {
    return this._map.size;
  }

  get(key) {
    return this._map.get(key);
  }

  has(key) {
    return this._map.has(key)
  }

  clear() {
    this._map.clear();
  }

  copy() {
    // Shallow copy only
    return new Map(this._map);
  }
}
