# Releasing Prestige

This document describes how to create a new release of Prestige.

## Prerequisites

- Push access to the main repository
- GitHub CLI (`gh`) installed (optional, for command-line release)

## One-Time Setup

### 1. Create gh-pages branch (for APT repository)

```bash
git checkout --orphan gh-pages
git rm -rf .
echo "# Prestige APT Repository" > README.md
git add README.md
git commit -m "Initialize gh-pages"
git push origin gh-pages
git checkout main
```

### 2. Enable GitHub Pages

1. Go to Settings → Pages
2. Set Source to "Deploy from a branch"
3. Select `gh-pages` branch and `/ (root)`
4. Save

### 3. Generate GPG Key (for signed APT packages)

```bash
# Generate a new GPG key
gpg --full-generate-key
# Choose: RSA and RSA, 4096 bits, no expiration
# Use email: prestige-release@users.noreply.github.com

# Export the private key
gpg --armor --export-secret-keys YOUR_KEY_ID > private.key

# Export the public key (this goes in gh-pages as gpg.key)
gpg --armor --export YOUR_KEY_ID > gpg.key
```

Add the private key as a GitHub secret named `GPG_PRIVATE_KEY`.

Upload `gpg.key` to the gh-pages branch:
```bash
git checkout gh-pages
# copy gpg.key here
git add gpg.key
git commit -m "Add GPG public key"
git push origin gh-pages
git checkout main
```

## Release Process

### 1. Update Version Numbers

Update the version in these files:

1. **`include/prestige/version.hpp`**:
   ```cpp
   #define PRESTIGE_VERSION_MAJOR X
   #define PRESTIGE_VERSION_MINOR Y
   #define PRESTIGE_VERSION_PATCH Z
   #define PRESTIGE_VERSION_STRING "X.Y.Z"
   ```

2. **`CMakeLists.txt`**:
   ```cmake
   project(prestige_uvs
     VERSION X.Y.Z
     ...
   )
   ```

3. **`Formula/prestige.rb`** (after release, update URL and sha256):
   ```ruby
   url "https://github.com/demajh/prestige/archive/refs/tags/vX.Y.Z.tar.gz"
   sha256 "ACTUAL_SHA256_OF_TARBALL"
   ```

### 2. Update Changelog (Optional)

If you maintain a CHANGELOG.md, update it with the new version's changes.

### 3. Commit Version Bump

```bash
git add -A
git commit -m "Bump version to X.Y.Z"
git push origin main
```

### 4. Create and Push Tag

```bash
# Create annotated tag
git tag -a vX.Y.Z -m "Release vX.Y.Z"

# Push tag to trigger release workflow
git push origin vX.Y.Z
```

### 5. Verify Release

1. Go to [GitHub Actions](https://github.com/demajh/prestige/actions)
2. Watch the "Release" workflow
3. Verify artifacts are uploaded to the [Releases page](https://github.com/demajh/prestige/releases)

### 6. Update Homebrew Formula

After the release is published:

1. Download the source tarball:
   ```bash
   curl -LO https://github.com/demajh/prestige/archive/refs/tags/vX.Y.Z.tar.gz
   ```

2. Calculate SHA256:
   ```bash
   shasum -a 256 vX.Y.Z.tar.gz
   ```

3. Update `Formula/prestige.rb` with the new URL and SHA256

4. If you have a separate homebrew-prestige tap repository, update it there

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

## Pre-releases

For pre-releases (alpha, beta, RC):

```bash
git tag -a vX.Y.Z-alpha.1 -m "Release vX.Y.Z-alpha.1"
git push origin vX.Y.Z-alpha.1
```

The release workflow will mark these as pre-releases automatically if the tag contains `-`.

## Manual Release (if automation fails)

If the GitHub Actions workflow fails, you can create a release manually:

1. Build the project locally for each platform
2. Create tarballs:
   ```bash
   DESTDIR=pkg cmake --install build
   tar -czvf prestige-X.Y.Z-platform.tar.gz -C pkg .
   ```
3. Go to GitHub → Releases → Draft a new release
4. Upload the tarballs
5. Write release notes
6. Publish

## Verifying Installation

After release, verify the installation works:

```bash
# From APT repository (Ubuntu/Debian)
curl -fsSL https://demajh.github.io/prestige/gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/prestige-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/prestige-archive-keyring.gpg] https://demajh.github.io/prestige stable main" | sudo tee /etc/apt/sources.list.d/prestige.list
sudo apt-get update
sudo apt-get install prestige
prestige_cli --help

# From .deb directly
curl -LO https://github.com/demajh/prestige/releases/download/vX.Y.Z/prestige_X.Y.Z-1_amd64.deb
sudo dpkg -i prestige_X.Y.Z-1_amd64.deb
sudo apt-get install -f
prestige_cli --help

# From binary tarball
curl -LO https://github.com/demajh/prestige/releases/download/vX.Y.Z/prestige-X.Y.Z-linux-x64.tar.gz
sudo tar -xzf prestige-X.Y.Z-linux-x64.tar.gz -C /
prestige_cli --help

# From Homebrew (after formula update)
brew update
brew install prestige
prestige_cli --help
```
