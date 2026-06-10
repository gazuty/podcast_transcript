# Security

## Threat model

`podcast-transcript` is a single-user, local CLI. It fetches URLs the user
nominates (a feed, an episode page, a direct MP3 link) and content those
URLs lead to (redirect targets, RSS-declared transcript files, links
scraped from HTML). The user is trusted; the remote servers and the
content they return are not.

## What is defended

- **URL schemes.** Every fetch entry point accepts only `http`/`https`, so
  `urllib` can never be steered at `file://` or other local handlers.
- **Redirects (SSRF).** The initial URL is user-chosen, but redirect
  targets are server-chosen — so every redirect hop is re-validated: the
  scheme must stay http(s) and the hostname must not resolve to a
  loopback, private (RFC 1918), link-local (cloud metadata), or otherwise
  non-public address. See `open_http` in `download.py`.
- **Response sizes.** All in-memory fetches (feed, transcript, page) cap
  the body *during* the read — an unbounded or multi-GB response is
  rejected after at most one 64 KiB chunk past the cap, not buffered
  whole. Audio downloads stream to disk and never buffer in memory.
- **XML.** Feeds containing `<!DOCTYPE`/`<!ENTITY` are rejected before
  parsing — stdlib `ElementTree` does not bound entity expansion (billion
  laughs), and no real podcast feed needs a DTD.
- **Content types.** Audio downloads must be served as `audio/*` or
  `application/octet-stream`; transcripts must be a caption type,
  `text/plain`, or octet-stream (`text/html` is refused), and the body
  must contain cue timestamps before it is accepted as a transcript.
- **Scraped links.** URLs harvested from episode-page HTML are dropped at
  scrape time unless they resolve to http(s), so a hostile page can't
  inject `file://`/`data:` URLs into the pipeline.
- **Bundled pack names.** `--corrections-pack` rejects path separators and
  leading dots, so a pack name can't traverse out of the package data dir.

## Accepted residual risks

- **DNS rebinding.** Redirect targets are resolved once for validation and
  again by `urllib` when connecting; an attacker controlling sub-second
  TTLs could pass the check and connect elsewhere. Closing this requires
  pinning the validated IP through the connection, which is not worth the
  complexity for a single-user local tool.
- **The initial URL is trusted.** If you point the tool at your own
  localhost service, it will fetch it — that's a feature of a local CLI,
  not a confused-deputy hole, because there is no third party supplying
  URLs except via redirects (which are validated).
- **Whisper/torch and the Anthropic SDK** are optional dependencies with
  their own supply chains; install the extras only if you use them.

## Reporting

Open a [GitHub issue](https://github.com/gazuty/podcast_transcript/issues),
or use GitHub's private vulnerability reporting on this repository if the
issue is sensitive.
