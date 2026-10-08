"""Hermes memory provider entry point (MemoryProvider): the plugin itself lives in the gemma_memory package."""

if __package__:  # loaded by Hermes as a plugin package (pytest imports this file bare)
    from .gemma_memory.provider import GemmaMemoryProvider

    def register(ctx) -> None:
        ctx.register_memory_provider(GemmaMemoryProvider())
