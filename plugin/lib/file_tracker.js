'use babel';

import {CompositeDisposable} from 'atom';
import path from 'path';

// The FileTracker keeps track of all TextEditors and their associated files.
// The purpose of the FileTracker is to let us retrieve the TextEditor(s)
// associated with a given file.

export default class FileTracker {
  constructor({
    projectRoot,
    onOpenFilesChange = () => {},
    onProjectFileChange = () => {},
    onProjectFileSave = () => {},
  }) {
    // We only care about files that fall under the project root
    this._projectRoot = projectRoot;

    // Event callbacks
    this._onProjectFileChange = onProjectFileChange;
    this._onProjectFileSave = onProjectFileSave;
    this._onOpenFilesChange = onOpenFilesChange;

    // A mapping of all observed (and not yet destroyed) TextEditors to a
    // disposable that is used to remove our event subscriptions from that
    // editor. The keys of this map are a set of all the TextEditors that we
    // know about.
    //
    // Map<TextEditor, Disposable>
    this._subscriptionsByEditor = new Map();

    // A mapping of a project file path to TextEditor(s) that are currently
    // displaying the file. This map is part of the editor index.
    //
    // Map<String, List<TextEditor>>
    this._editorsByFilePath = new Map();

    // Stores the inverse mapping of the previous map. This map is part of
    // the editor index.
    //
    // Map<TextEditor, String>
    this._filePathByEditor = new Map();

    // A set of all TextEditors that are displaying a project file. This set
    // is part of the editor index.
    this._editorsShowingProjectFiles = new Set();

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

    this._editorsShowingProjectFiles.clear();
    this._editorsShowingProjectFiles = null;
  }

  getTextEditorsFor(filePath) {
    if (!this._editorsByFilePath.has(filePath)) {
      return [];
    }
    return this._editorsByFilePath.get(filePath);
  }

  _onNewEditor(editor) {
    // Subscribe to relevant events on the TextEditor
    const subs = new CompositeDisposable();
    subs.add(editor.onDidChangePath(this._onEditorPathChange.bind(this, editor)));
    subs.add(editor.onDidStopChanging(this._onEditorChange.bind(this, editor)));
    subs.add(editor.onDidSave(this._onEditorSave.bind(this, editor)));
    subs.add(editor.onDidDestroy(this._onEditorDistroy.bind(this, editor)));
    this._subscriptionsByEditor.set(editor, subs);

    if (this._addEditorToIndexIfNeeded(editor)) {
      this._onOpenFilesChange();
    }
  }

  _onEditorPathChange(editor) {
    const removalChange = this._removeEditorFromIndexIfNeeded(editor);
    const additionChange = this._addEditorToIndexIfNeeded(editor);
    if (removalChange || additionChange) {
      this._onOpenFilesChange();
    }
  }

  _onEditorSave(editor) {
    if (!this._editorsShowingProjectFiles.has(editor)) {
      return;
    }
    this._onProjectFileSave(editor);
  }

  _onEditorChange(editor) {
    if (!this._editorsShowingProjectFiles.has(editor)) {
      return;
    }
    this._onProjectFileChange(editor);
  }

  _onEditorDistroy(editor) {
    const indexChanged = this._removeEditorFromIndexIfNeeded(editor);
    this._subscriptionsByEditor.get(editor).dispose();
    this._subscriptionsByEditor.delete(editor);

    if (indexChanged) {
      this._onOpenFilesChange();
    }
  }

  _toProjectRelativePath(candidateFilePath) {
    if (!candidateFilePath.startsWith(this._projectRoot)) {
      return null;
    }
    return path.relative(this._projectRoot, candidateFilePath);
  }

  _addEditorToIndexIfNeeded(editor) {
    // Abort if the editor is already in our index
    if (this._editorsShowingProjectFiles.has(editor)) {
      return false;
    }

    // Abort if the editor is not displaying a project file
    const editorPathAbsolute = editor.getPath();
    const projectFilePath = this._toProjectRelativePath(editorPathAbsolute);
    if (projectFilePath == null) {
      return false;
    }

    this._editorsShowingProjectFiles.add(editor);
    if (!this._editorsByFilePath.has(projectFilePath)) {
      this._editorsByFilePath.set(projectFilePath, []);
    }
    this._editorsByFilePath.get(projectFilePath).push(editor);
    this._filePathByEditor.set(editor, projectFilePath);

    return true;
  }

  _removeEditorFromIndexIfNeeded(editor) {
    // Abort if the editor is not in our index
    if (!this._editorsShowingProjectFiles.has(editor)) {
      return false;
    }

    this._editorsShowingProjectFiles.delete(editor);
    const filePath = this._filePathByEditor.get(editor);
    this._filePathByEditor.delete(editor);
    this._editorsByFilePath.set(
      filePath,
      this._editorsByFilePath.get(filePath)
        .filter(candidateEditor => editor !== candidateEditor),
    );

    return true;
  }
}
