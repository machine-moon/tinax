# Standard Library

`tinax.stdlib` converts explicit argument sequences into caller-owned configuration and creates isolated stream loggers. It never reads `sys.argv`, calls `logging.basicConfig`, mutates root handlers, or registers named loggers globally.

`parse_config` passes a shallow read-only mapping to the factory; mutable values produced by custom argparse actions remain caller-owned. Its `argv` excludes the executable name, so an Abseil entry point passes `argv[1:]`.

Tinax does not provide an Abseil domain. Grain executables still enter through `absl.app.run` because that upstream worker lifecycle requires parsed Abseil flags.

```python
import logging
import sys

from tinax.stdlib import make_stream_logger

logger = make_stream_logger(
    "trainer",
    level=logging.INFO,
    stream=sys.stderr,
    format_string="%(levelname)s %(message)s",
)
```

::: tinax.stdlib.parse_config

::: tinax.stdlib.make_stream_logger
