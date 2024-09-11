# Maintainers notes

> Workflow information mainly relevant to maintainers

## PR review workflow

- Submit a PR (mark as draft if your feature isn't ready yet, but still want to share your work)
- Request PR to be reviewed by at least one maintainer. Other users are also welcome to submit reviews on PRs.
- Implement or discuss suggested edits
- Once PR is approved:
  - Original author merges the PR (if original author has sufficient permissions)
  - Wait for maintainer to merge
  - If more edits are required: Implement edits and re-request review if changes are significant
- Close linked issue

---

- If PR is automated (i.e., from dependabot or similar), maintainer can review and merge.

## Release checklist

- Go to GitHub, draft new release. Enter name of version and "create new tag" if it doesn't already exist. Click "Generate Release Notes". Currate release notes as needed. Look at a previous version release to match the format (title, header, section organisation etc.)
- Go to [conda-forge/parcels-feedstock](https://github.com/conda-forge/parcels-feedstock), create new issue with `@conda-forge-admin, please update version`. This will prompt a build, otherwise there can be a delay in the build.
  - Approve PR and merge on green
- Update version in `CITATION.cff` file
- Check "publish to PyPI" workflow succeeded
- Update oceanparcels.org
  - Parcels development status
  - Check feature tiles
  - Check for broken links on oceanparcels using [this tracking issue](https://github.com/OceanParcels/oceanparcels_website/issues/85)
- (once package is available on conda) Re-build the Binder
