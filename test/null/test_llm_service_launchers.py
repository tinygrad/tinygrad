import os, pathlib, re, subprocess, unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "extra/llm_service"


class TestLLMServiceLaunchers(unittest.TestCase):
  def command(self, script: str, extra_env: dict|None = None) -> str:
    env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], "VISION_MMPROJ": ""}
    if extra_env is not None: env.update(extra_env)
    return subprocess.check_output([SCRIPTS / script, "--print-command"], cwd=ROOT, text=True, env=env, stderr=subprocess.STDOUT)

  def test_scripts_parse_and_help(self):
    self.assertTrue(all(p.stat().st_mode & 0o111 for p in SCRIPTS.glob("*.sh")))
    for script in ("qwen_6x7900xtx.sh", "glm_flash_6x7900xtx.sh", "glm_14xr9700.sh"):
      subprocess.run(["bash", "-n", SCRIPTS / script], check=True)
      help_text = subprocess.check_output([SCRIPTS / script, "--help"], text=True, stderr=subprocess.STDOUT)
      self.assertIn("--print-command", help_text)

  def test_qwen_v31_production_environment(self):
    cmd = self.command("qwen_6x7900xtx.sh")
    for required in (
      "DEV=AMD",
      "AM_MEC_PIPES=2",
      "AM_POWER_LIMIT=250",
      "SEQ_ROWS=4310016",
      "SEQ_STASH_GB=46",
      "QSA_KV_BITS4_LAYERS=39\\,43\\,47\\,48",
      "VISION_MAX_IMAGES=1024",
      "VISION_URL_CACHE_MB=512",
      "VISION_MM" + "PROJ=/weights/Qwen3.8-Flash-Next-UD-IQ4_XS-V31/mmproj/mmproj-F16.gguf",
      "--shard 6",
      "--max_context 262144",
      "--serve 8000",
    ):
      self.assertIn(required, cmd)
    self.assertIn("Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf", cmd)
    self.assertNotIn("00001-of-00003.gguf", cmd)
    self.assertNotIn("AM_VM_RESET=1", cmd)
    self.assertNotIn("SEQ_STASH_ENTRIES=", cmd)
    pokemon = self.command("qwen_6x7900xtx.sh", {"SPEC_DRAFTS": "3", "SERVE_RENDER_CACHE": "1", "SEQ_CKPT_TAIL": "1",
                                                     "VISION_BUCKETS": "384"})
    for required in ("SPEC_DRAFTS=3", "SERVE_RENDER_CACHE=1", "SEQ_CKPT_TAIL=1", "VISION_BUCKETS=384", "TOP_K=20"):
      self.assertIn(required, pokemon)

  def test_glm_production_environment_and_power_guard(self):
    cmd = self.command("glm_14xr9700.sh")
    self.assertIn("VISION_MMPROJ=", cmd)
    for required in (
      "DEV=PCI+AMD",
      "VFIO=1",
      "SPEC_DRAFTS_BY_SIZE=1:2\\,2:2\\,4:3\\,8:3\\,16:3",
      "SEQ_GATHER_UNION=65536",
      "SEQ_STASH_GB=0",
      "PREFILL_CHUNK=4032",
      "PREFILL_SIZES=1344\\,2688",
      "MODEL_ALIASES=zai-org/GLM-5.3",
      "--shard 14",
      "--max_context 262144",
      "--serve 8000",
    ):
      self.assertIn(required, cmd)
    self.assertIn("PENDING_POWER_VERIFICATION", cmd)
    self.assertIn("GLM-5.3-UD-IQ4_XS-00001-of-00009.gguf", cmd)
    self.assertNotIn("PREFILL_MB1_ROWS=", cmd)
    profile = self.command("glm_14xr9700.sh", {"GLM_KV": "q4", "MAX_CONTEXT": "1048576", "SEQ_ROWS": "1048576",
                                               "SEQ_SLOTS": "16", "SEQ_MAX_ACTIVE": "16", "P2P_GLM5_OFF": "1",
                                               "FREQS_HOST": "1", "AMD_SCRATCH_MAX": "64", "SEQ_COPY_SCRATCH": "1",
                                               "LOAD_DEFER": "1", "AM_PERF_LEVEL": "high", "AM_CLOCKS": "GFXCLK=2000",
                                               "CLOCK_FAST": "GFXCLK=2000", "CLOCK_MANY": "GFXCLK=2000",
                                               "CLOCK_PREFILL": "GFXCLK=2000"})
    for required in ("GLM_KV=q4", "--max_context 1048576", "SEQ_ROWS=1048576", "SEQ_SLOTS=16", "SEQ_MAX_ACTIVE=16",
                       "P2P_GLM5_OFF=1", "FREQS_HOST=1", "AMD_SCRATCH_MAX=64", "SEQ_COPY_SCRATCH=1", "LOAD_DEFER=1",
                       "AM_PERF_LEVEL=high", "AM_CLOCKS=GFXCLK=2000", "CLOCK_FAST=GFXCLK=2000",
                       "CLOCK_MANY=GFXCLK=2000", "CLOCK_PREFILL=GFXCLK=2000"):
      self.assertIn(required, profile)
    failed = subprocess.run([SCRIPTS / "glm_14xr9700.sh", "start"], cwd=ROOT, text=True, capture_output=True)
    self.assertEqual(failed.returncode, 2)
    self.assertIn("PENDING", failed.stderr)
    self.assertIn("210 W is not safe", failed.stderr)

  def test_qwen_power_guard(self):
    proc = subprocess.run(
      [SCRIPTS / "qwen_6x7900xtx.sh", "start"], env={**os.environ, "AM_POWER_LIMIT": "251"}, cwd=ROOT, text=True, capture_output=True
    )
    self.assertEqual(proc.returncode, 2)
    self.assertIn("225", proc.stderr)

  def test_glm_flash_d34_environment_and_power_guard(self):
    cmd = self.command("glm_flash_6x7900xtx.sh")
    for required in (
      "DEV=AMD", "AM_POWER_LIMIT=225", "BATCH_SLOTS=8", "SEQ_KV4=1", "SEQ_ROWS=3916800",
      "SEQ_SIZES=1\\,2\\,4\\,8\\,12\\,13\\,18", "SEQ_STASH_GB=0", "SEQ_IDX_SHARD=1",
      "PREFILL_CHUNK=768", "PREFILL_SIZES=192\\,768", "PREFILL_MB1_ROWS=1200", "PREFILL_SLICE=1536",
      "SEQ_STARVE_ALT=1", "--shard 6", "--max_context 524288",
    ):
      self.assertIn(required, cmd)
    self.assertIn("GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf", cmd)
    self.assertNotIn("SPEC_DRAFTS=", cmd)  # D34 inherits GLM's default of two drafts.
    self.assertNotIn("AM_VM_RESET=1", cmd)
    self.assertIn("VISION_MMPROJ=/home/runner/glmvis/mmproj-F16.gguf", cmd)
    self.assertIn("VISION_BUCKETS=576", cmd)
    for optional in ("VISION_SLICE_MS", "VISION_MAX_TOKENS", "VISION_MAX_IMAGES", "VISION_CACHE_MB",
                     "VISION_URL_CACHE_MB", "SERVE_RENDER_CACHE"):
      self.assertNotIn(optional + "=", cmd)  # The benchmark default, not the Pokemon G3 opt-in.
    picture = self.command("glm_flash_6x7900xtx.sh", {"VISION_SLICE_MS": "200", "VISION_MAX_TOKENS": "1024",
                                                       "VISION_MAX_IMAGES": "2048", "VISION_CACHE_MB": "2048",
                                                       "VISION_URL_CACHE_MB": "512", "SERVE_RENDER_CACHE": "1"})
    for required in ("VISION_MMPROJ=/home/runner/glmvis/mmproj-F16.gguf", "VISION_SLICE_MS=200",
                     "VISION_MAX_TOKENS=1024", "VISION_MAX_IMAGES=2048", "VISION_CACHE_MB=2048",
                     "VISION_URL_CACHE_MB=512", "VISION_BUCKETS=576", "SERVE_RENDER_CACHE=1"):
      self.assertIn(required, picture)
    failed = subprocess.run(
      [SCRIPTS / "glm_flash_6x7900xtx.sh", "start"], env={**os.environ, "AM_POWER_LIMIT": "251"},
      cwd=ROOT, text=True, capture_output=True,
    )
    self.assertEqual(failed.returncode, 2)
    self.assertIn("225", failed.stderr)

  def test_c5_null_harness_keeps_benchmark_defaults(self):
    body = (ROOT / "test/null/production_c5_warmup.sh").read_text()
    self.assertIn("export VISION_MMPROJ VISION_BUCKETS=576", body)
    for optional in ("VISION_SLICE_MS", "VISION_MAX_TOKENS", "VISION_MAX_IMAGES", "VISION_CACHE_MB",
                     "VISION_URL_CACHE_MB", "SERVE_RENDER_CACHE"):
      self.assertNotIn(optional + "=", body)
    self.assertIn('export "$_extra"', body)
    for name in ("VISION_SLICE_MS", "VISION_MAX_TOKENS", "VISION_MAX_IMAGES", "VISION_CACHE_MB",
                 "VISION_URL_CACHE_MB", "SERVE_RENDER_CACHE"):
      self.assertIn(name, body)

  def test_launch_files_do_not_set_folded_controls(self):
    obsolete = (
      "FLAG_AR_POOLN", "LOAD_APART", "MICRO_BATCHES", "QSA_IDX_SPLIT", "QSA_KV_BITS", "QSA_KV_ROT",
      "QSA_KV_SPLIT", "RING_MAXW_DEC", "RING_SEL", "VISION_KERNELS", "FLAG_AR_MAXN", "IQ2S_K256",
      "MLA_SEQ_SHARD", "RING_MAXW", "RING_RS_LIKE_COPIES", "SEQ_WARMUP_ONLY", "WARMUP_PARALLEL",
      "GLM_ATTN_GRP", "GLM_SEQ",
    )
    files = [SCRIPTS / name for name in ("qwen_6x7900xtx.sh", "glm_flash_6x7900xtx.sh", "glm_14xr9700.sh")]
    files += [ROOT / "test/null" / f"production_{name}_warmup.sh" for name in ("c2", "c5", "c9")]
    for path in files + [ROOT / "RUNNING.md"]:
      body = path.read_text()
      for name in obsolete:
        self.assertNotRegex(body, rf"(?<![A-Z0-9_]){re.escape(name)}=", f"{path}: folded control {name}")

  def test_manifests_match_runbook(self):
    runbook = (ROOT / "RUNNING.md").read_text()
    for manifest, count in (("qwen-v31.sha256", 6), ("glm-flash-iq3xxs.sha256", 4), ("glm53-iq4xs.sha256", 9)):
      lines = (SCRIPTS / manifest).read_text().strip().splitlines()
      self.assertEqual(len(lines), count)
      for line in lines:
        digest, path = line.split()
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertIn(path, runbook)
        self.assertEqual(runbook.count(digest), 1, path)

  def test_runbook_and_null_warmups_are_self_contained(self):
    runbook = (ROOT / "RUNNING.md").read_text()
    self.assertNotIn("operations `results/", runbook)
    self.assertNotIn("tools/launch/", runbook)
    self.assertNotIn("results/w-gpf/", runbook)
    for service in ("c2", "c5", "c9"):
      script = ROOT / f"test/null/production_{service}_warmup.sh"
      harness = ROOT / f"test/null/production_{service}_warmup.py"
      self.assertTrue(script.stat().st_mode & 0o111)
      self.assertTrue(harness.is_file())
      subprocess.run(["bash", "-n", script], check=True)
      self.assertIn(f"test/null/production_{service}_warmup.sh", runbook)


if __name__ == "__main__":
  unittest.main()
