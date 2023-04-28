# How to release a new version of DeepView.Profile
- Go to Github repo and run the action `build-and-publish-new-version`. You will be prompted to specify the version number.

- This runs a GitHub Action that will take the following steps:
   1. Fetches the repo and its dependencies
   2. Creates a release branch
   3. Updates the version number to the user-specified version by updating the pyproject.toml
   4. Commits the changes and tag the commit with the version number
   5. Builds the Python build artifacts
   7. Publishes a release to Github
   8. Create a PR to merge back into main
   9. Publishes to Test PyPI
   10. Publishes to PyPI
   
- The action `build-and-publish-new-version` is defined under `.github/workflows/build-and-publish-new-versionyaml`

- This release process follows the release process outlined in [OneFlow](https://www.endoflineblog.com/oneflow-a-git-branching-model-and-workflow).