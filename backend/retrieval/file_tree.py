"""File tree builder from indexed chunk metadata."""


def build_file_tree(chunks_metadata: list[dict]) -> dict:
    """Build a nested file tree structure from chunk metadata."""
    tree = {}

    file_paths = set()
    for meta in chunks_metadata:
        file_paths.add(meta.get("file_path", ""))

    for path in sorted(file_paths):
        if not path:
            continue
        parts = path.split("/")
        node = tree
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            node = node[part]
        node[parts[-1]] = None  # leaf file

    return _format_tree(tree)


def _format_tree(tree: dict, prefix: str = "") -> list[dict]:
    """Convert nested dict to flat list with type and depth info."""
    items = []
    entries = sorted(tree.items(), key=lambda x: (x[1] is not None, x[0]))

    for i, (name, subtree) in enumerate(entries):
        is_last = i == len(entries) - 1

        if subtree is None:
            items.append({
                "name": name,
                "type": "file",
                "path": f"{prefix}{name}" if not prefix else f"{prefix}/{name}",
            })
        else:
            path = f"{prefix}{name}" if not prefix else f"{prefix}/{name}"
            items.append({
                "name": name,
                "type": "directory",
                "path": path,
                "children": _format_tree(subtree, path),
            })

    return items
