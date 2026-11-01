def recursive_chunk(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> list[str]:

    separators = ["\n\n", "\n", ". ", " ", ""]

    def split_text(text: str, separators: list[str]) -> list[str]:

        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        if not separators:
            return [
                text[i:i + chunk_size]
                for i in range(0, len(text), chunk_size)
            ]

        separator = separators[0]

        if separator == "":
            pieces = list(text)
        else:
            pieces = text.split(separator)

        result = []
        current = ""

        for piece in pieces:
            candidate = (
                current + separator + piece
                if current else piece
            )

            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)

                if len(piece) > chunk_size:
                    result.extend(
                        split_text(piece, separators[1:])
                    )
                    current = ""
                else:
                    current = piece

        if current:
            result.append(current)

        return result

    pieces = split_text(text, separators)

    if overlap == 0:
        return pieces

    chunks = []
    current = ""

    for piece in pieces:
        candidate = (
            current + " " + piece
            if current else piece
        )

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)

            # Preserve the tail of the previous chunk
            current = current[-overlap:] + " " + piece if current else piece

            if len(current) > chunk_size:
                current = current[-chunk_size:]

    if current:
        chunks.append(current)

    return [chunk.strip() for chunk in chunks if chunk.strip()]