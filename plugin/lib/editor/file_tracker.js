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
    this._editorsByFilePath = new Map();

    // Stores all project TextEditors that have buffers with unsaved changes.
    // This set is part of the editor index.
    //
    // Set<TextEditor>
    this._modifiedEditors = new Set();

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

    this._modifiedEditors.clear();
    this._modifiedEditors = null;
  }

  getTextEditorsFor(filePath) {
    if (!this._editorsByFilePath.has(filePath)) {
      return [];
    }
    return this._editorsByFilePath.get(filePath);
  }

  isProjectModified() {
    return this._modifiedEditors.size > 0;
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
    if (!this._filePathByEditor.has(editor)) {
      return;
    }

    const prevProjectModified = this.isProjectModified();
    if (editor.isModified()) {
      this._modifiedEditors.add(editor);
    } else {
      this._modifiedEditors.delete(editor);
    }

    if (prevProjectModified !== this.isProjectModified()) {
      process.nextTick(this._onProjectModifiedChange);
    }
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
    if (!this._editorsByFilePath.has(projectFilePath)) {
      this._editorsByFilePath.set(projectFilePath, []);
    }
    this._editorsByFilePath.get(projectFilePath).push(editor);
    this._filePathByEditor.set(editor, projectFilePath);

    const callbacks = [this._onOpenFilesChange];
    if (!editor.isModified()) {
      return callbacks;
    }

    // If the editor has unsaved changes, keep track of it
    const prevProjectModified = this.isProjectModified();
    this._modifiedEditors.add(editor);
    if (prevProjectModified !== this.isProjectModified()) {
      callbacks.push(this._onProjectModifiedChange);
    }
    return callbacks;
  }

  _removeEditorFromIndexIfNeeded(editor) {
    // Abort if the editor is not in our index
    if (!this._filePathByEditor.has(editor)) {
      return [];
    }

    const filePath = this._filePathByEditor.get(editor);
    this._filePathByEditor.delete(editor);
    this._editorsByFilePath.set(
      filePath,
      this._editorsByFilePath.get(filePath)
        .filter(candidateEditor => editor !== candidateEditor),
    );

    const callbacks = [this._onOpenFilesChange];

    const prevProjectModified = this.isProjectModified();
    this._modifiedEditors.delete(editor);
    if (prevProjectModified !== this.isProjectModified()) {
      callbacks.push(this._onProjectModifiedChange);
    }

    return callbacks;
  }
}
