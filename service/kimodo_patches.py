"""patches so kimodo runs with a local llama copy."""

import os

HUB_ID = "meta-llama/Meta-Llama-3-8B-Instruct"


def _same_path(a, b):
    try:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))
    except (TypeError, ValueError):
        return False


# workaround for nv-tlabs/kimodo issue #57, drop it once that's fixed
def patch_empty_constraint_sets():
    try:
        from kimodo.motion_rep.conditioning import ConditionWriters
    except ImportError:
        return
    if getattr(ConditionWriters.gather, "_kimodo_patched", False):
        return
    orig_gather = ConditionWriters.gather

    def gather(self, fname):
        entries = self.index_dict.get(fname) if hasattr(self.index_dict, "get") else None
        if entries and all(len(e) == 0 for e in entries):
            return None
        return orig_gather(self, fname)

    gather._kimodo_patched = True
    ConditionWriters.gather = gather


def apply(llama_dir=None, template=True):
    """install the patches, template=False skips patch 2"""
    patch_empty_constraint_sets()
    llama_dir = llama_dir or os.environ.get("KIMODO_LLAMA_DIR")
    if not llama_dir or not os.path.isfile(os.path.join(llama_dir, "config.json")):
        raise RuntimeError(f"KIMODO_LLAMA_DIR does not point at a Llama-3 model folder: {llama_dir!r}")

    import transformers.modeling_utils as mu

    if not getattr(mu.maybe_load_adapters, "_kimodo_patched", False):
        orig_load = mu.maybe_load_adapters

        def maybe_load_adapters(*args, **kwargs):
            adapter_path, base, adapter_kwargs = orig_load(*args, **kwargs)
            if base == HUB_ID:
                base = llama_dir
            return adapter_path, base, adapter_kwargs

        maybe_load_adapters._kimodo_patched = True
        mu.maybe_load_adapters = maybe_load_adapters

    if template:
        from kimodo.model.llm2vec.llm2vec import LLM2Vec

        if not getattr(LLM2Vec.prepare_for_tokenization, "_kimodo_patched", False):
            orig_prepare = LLM2Vec.prepare_for_tokenization

            def prepare_for_tokenization(self, text):
                if _same_path(self.model.config._name_or_path, llama_dir):
                    return "<|start_header_id|>user<|end_header_id|>\n\n" + text.strip() + "<|eot_id|>"
                return orig_prepare(self, text)

            prepare_for_tokenization._kimodo_patched = True
            LLM2Vec.prepare_for_tokenization = prepare_for_tokenization
