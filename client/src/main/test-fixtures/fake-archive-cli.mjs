#!/usr/bin/env node
import { appendFileSync } from "node:fs";

const args = process.argv.slice(2);

// Mirror the real CLI: a `version` subcommand exists, `--version` does not.
if (args[0] === "version") {
  process.stdout.write("archive 1.2.0\n");
  process.exit(0);
}
if (args.includes("--version")) {
  process.stderr.write('archive: unknown command "--version"; run archive help\n');
  process.exit(1);
}

if (process.env.CADENCE_FAKE_ARCHIVE_FAIL) {
  process.stderr.write("fake archive failure\n");
  process.exit(1);
}

const logPath = process.env.CADENCE_FAKE_ARCHIVE_LOG;
if (logPath) appendFileSync(logPath, `${JSON.stringify(args)}\n`);

process.stdout.write("Jotted.\n");
