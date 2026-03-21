# Momo-Kibidango Installation Troubleshooting Guide

**Version:** 1.0  
**Date:** March 20, 2026  
**For:** Phase 1 Script-Based Installation

---

## Quick Diagnosis

Run the validation script first:

```bash
./validate-installation.sh
```

This will check all components and report any issues.

---

## Common Issues

### 1. Python Version Error

**Error Message:**
```
❌ Python 3.10+ required (found 3.9.x)
```

**Solution:**
```bash
# Check installed Python versions
python3 --version
python3.11 --version  # or python3.12, etc.

# Install via Homebrew (macOS)
brew install python@3.11

# Install via apt (Linux)
sudo apt-get install python3.11

# Then re-run install script
./install.sh
```

**Note:** The installer requires Python 3.10 or later due to dependencies on modern type hints and asyncio features.

---

### 2. Insufficient Disk Space

**Error Message:**
```
❌ Insufficient disk space (need 20GB)
```

**Solution:**
1. Check available space:
   ```bash
   df -h ~
   ```

2. Free up space if needed:
   ```bash
   # Clear cache
   rm -rf ~/.cache/pip
   
   # Clear Homebrew cache (macOS)
   brew cleanup
   
   # Clear old conda environments (if using conda)
   conda clean --all
   ```

3. Consider smaller installation:
   - Skip large models initially
   - Install only 2-model (not 3-model) configuration

---

### 3. Virtual Environment Already Exists

**Warning Message:**
```
⚠ Virtual environment already exists
Remove with: rm -rf /Users/xxx/.momo-kibidango/venv
```

**Solutions:**

**Option A:** Clean reinstall
```bash
./uninstall.sh
./install.sh
```

**Option B:** Keep existing installation
```bash
# Just re-run install to update dependencies
./install.sh
```

**Option C:** Manual cleanup
```bash
rm -rf ~/.momo-kibidango/venv
./install.sh
```

---

### 4. pip/setuptools Version Conflicts

**Error Message:**
```
ERROR: Could not find a version that satisfies the requirement...
```

**Solution:**
```bash
# Upgrade pip, setuptools, wheel
python3 -m pip install --upgrade pip setuptools wheel

# Then retry
./install.sh
```

Or manually:
```bash
source ~/.momo-kibidango/venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch transformers pydantic numpy tqdm pyyaml
```

---

### 5. Git Repository Detection Warning

**Warning Message:**
```
⚠ Not a git repository. Using update mode: dependencies only
```

**Context:** Occurs during `./update.sh` on a cloned but detached repo.

**Solution:**
```bash
# This is normal if you downloaded the script directly
# Updates will still work - just dependency updates, not repo pulls

# If you want full git support:
cd ~/momo-kibidango
git init
git add .
git remote add origin https://github.com/rdreilly58/momo-kibidango.git
git branch --set-upstream-to=origin/main main
```

---

### 6. macOS-Specific: Disk Access Denied

**Error Message:**
```
Permission denied: /Users/xxx/.momo-kibidango
```

**Solution:**
```bash
# Give full disk access to Terminal
# System Settings → Privacy & Security → Full Disk Access → Add Terminal

# Or use sudo (not recommended)
sudo ./install.sh
```

**Better:** Grant Terminal full disk access via System Settings.

---

### 7. Linux: Missing Build Tools

**Error Message:**
```
error: Microsoft Visual C++ 14.0 or greater is required
```

**Solution (Linux):**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential python3-dev

# Fedora/RHEL
sudo dnf groupinstall "Development Tools"
sudo dnf install python3-devel
```

**Solution (macOS):**
```bash
xcode-select --install
```

---

### 8. Validation Tests Fail

**After Installation:**
```bash
# Run validation
./validate-installation.sh
```

**If tests fail:**

**Option A:** Check dependencies manually
```bash
source ~/.momo-kibidango/venv/bin/activate
python3 -c "import torch; print(torch.__version__)"
python3 -c "import transformers; print(transformers.__version__)"
```

**Option B:** Reinstall from scratch
```bash
./uninstall.sh
rm -rf ~/.cache/pip ~/.cache/huggingface
./install.sh
```

**Option C:** Check install log
```bash
tail -f ~/momo-kibidango/install.log
```

---

### 9. Hugging Face Model Download Fails

**Error Message:**
```
OSError: Can't find 'config.json' in Qwen/Qwen2-7B
```

**Causes:**
- Network connectivity issue
- Hugging Face API rate limit
- Missing model access permission

**Solutions:**

**Check connectivity:**
```bash
curl -I https://huggingface.co
```

**Set HF token (if needed):**
```bash
huggingface-cli login
# Enter your access token from https://huggingface.co/settings/tokens
```

**Manual model download:**
```bash
source ~/.momo-kibidango/venv/bin/activate
huggingface-cli download Qwen/Qwen2-7B --local-dir ~/.momo-kibidango/models/qwen2-7b
```

---

### 10. Memory Issues During Installation

**Symptoms:**
- Installation hangs
- "out of memory" messages
- System becomes very slow

**Solutions:**

**Option A:** Reduce pip processes
```bash
source ~/.momo-kibidango/venv/bin/activate
pip install --no-cache-dir torch transformers pydantic numpy tqdm pyyaml
```

**Option B:** Install one package at a time
```bash
source ~/.momo-kibidango/venv/bin/activate
for pkg in torch transformers pydantic numpy tqdm pyyaml; do
  pip install "$pkg"
done
```

**Option C:** Use pre-built wheels
```bash
# macOS M1/M2/M3 (Apple Silicon)
pip install torch torchvision torchaudio -i https://download.pytorch.org/whl/nightly/cpu
```

---

## Validation Checklist

After installation, verify:

- [ ] Virtual environment created: `ls ~/.momo-kibidango/venv/`
- [ ] Configuration exists: `cat ~/.momo-kibidango/config/config.yaml`
- [ ] Models directory ready: `ls ~/.momo-kibidango/models/`
- [ ] Dependencies installed: `./validate-installation.sh`
- [ ] Python imports work: `source ~/.momo-kibidango/venv/bin/activate && python3 -c 'import torch'`
- [ ] No error log: `tail ~/momo-kibidango/install.log`

---

## Uninstall & Cleanup

**Uninstall (keep cache):**
```bash
./uninstall.sh
```

**Complete cleanup (remove cache too):**
```bash
./uninstall.sh
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/pip
```

**Verify removal:**
```bash
ls ~/.momo-kibidango  # Should not exist
```

---

## Getting Help

### Check Logs

```bash
# Installation log
tail -f ~/momo-kibidango/install.log

# Recent messages
grep "ERROR\|FAILED" ~/momo-kibidango/install.log
```

### Debug Information

Gather this before reporting issues:

```bash
# System info
uname -a
python3 --version

# Installation details
ls -la ~/.momo-kibidango/
cat ~/.momo-kibidango/config/config.yaml

# Recent errors
tail -50 ~/momo-kibidango/install.log
```

### Report Issues

When reporting problems, include:
1. OS and version
2. Python version
3. Full error message
4. Output of validation script
5. Last 50 lines of install.log

---

## FAQ

### Q: Can I install to a custom location?

**A:** Not yet. Current version uses `~/.momo-kibidango`. Custom paths coming in v1.1.

### Q: Can I skip model downloads?

**A:** Edit `install.sh` to comment out the model download section, but core functionality requires at least one model.

### Q: How do I update after installation?

**A:** Run the update script:
```bash
./update.sh
```

### Q: Can I use different models?

**A:** Yes, edit `~/.momo-kibidango/config/config.yaml` with your model paths.

### Q: Does it work on Windows?

**A:** WSL2 (Windows Subsystem for Linux 2) works fine. Native Windows PowerShell coming soon.

### Q: Why Python 3.10+?

**A:** Required for:
- `asyncio` features
- Type hint syntax (PEP 604: `int | str`)
- `match` statements (pattern matching)

### Q: Can I use Conda instead of venv?

**A:** Yes:
```bash
conda create -n momo-kibidango python=3.10
conda activate momo-kibidango
pip install torch transformers pydantic numpy tqdm pyyaml
```

Then configure paths in config.yaml.

---

## Performance Tips

After successful installation:

1. **First run is slow** - Models are cached, subsequent runs are faster
2. **Use CPU caching** - Keep models in `~/.momo-kibidango/models/`
3. **Monitor disk** - Cached models use ~15-20GB
4. **Check logs** - Review `install.log` for performance hints

---

## Contact & Support

- **GitHub Issues:** https://github.com/rdreilly58/momo-kibidango/issues
- **Documentation:** https://github.com/rdreilly58/momo-kibidango/tree/main/docs
- **Design Docs:** Read [MOMO_KIBIDANGO_INSTALLATION_DESIGN.md](docs/MOMO_KIBIDANGO_INSTALLATION_DESIGN.md)

---

*Last Updated: March 20, 2026*  
*For Phase 1 Script-Based Installation*
