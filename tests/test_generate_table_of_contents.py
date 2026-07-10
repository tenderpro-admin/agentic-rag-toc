from document_ingestion.generate_table_of_contents import DocumentTOCParser


def test_split_table_item_repeats_compact_grouped_header_for_llm_context() -> None:
    parser = DocumentTOCParser()
    table_text = "\n".join(
        [
            "| | Twelve Months Ended June 30, 2022 | Twelve Months Ended June 30, 2022 | Twelve Months Ended June 30, 2023 | Twelve Months Ended June 30, 2023 |",
            "|---|---|---|---|---|",
            "| ($ million) | Flexibles | Total | Flexibles | Total |",
            "| Net income attributable to Amcor |  | 805 |  | 1,048 |",
            "| Tax expense |  | 300 |  | 193 |",
        ]
    )
    lines = table_text.splitlines()
    compact_header = parser._build_compact_repeated_table_header(lines[0], lines[2])[0]
    first_data_row = "| Net income attributable to Amcor |  | 805 |  | 1,048 |"
    max_chars = len(compact_header) + len(first_data_row) + 5

    assert len("\n".join(lines[:4])) > max_chars
    assert len("\n".join([compact_header, first_data_row])) <= max_chars

    parts = parser._split_table_item(
        {"text": table_text, "label": "table"},
        max_chars=max_chars,
    )

    assert len(parts) == 2
    for part in parts:
        lines = [line for line in part["text"].splitlines() if line.strip()]
        assert lines[0] == (
            "| ($ million) | Twelve Months Ended June 30, 2022 Flexibles | "
            "Twelve Months Ended June 30, 2022 Total | "
            "Twelve Months Ended June 30, 2023 Flexibles | "
            "Twelve Months Ended June 30, 2023 Total |"
        )
        assert len(part["text"]) <= max_chars


def test_split_table_item_keeps_context_rows_with_following_numeric_rows() -> None:
    parser = DocumentTOCParser()
    table_text = "\n".join(
        [
            "| Five-Year Financial Summary | Five-Year Financial Summary | Five-Year Financial Summary |",
            "|---|---|---|",
            "| In millions, except per share amounts | 2018 | 2017 |",
            "| Per common share data: | | |",
            "| Basic earnings (loss) per common share: | | |",
            "| Income (loss) from continuing operations attributable to CVS Health | $ (0.57) | $ 6.48 |",
            "| Net income (loss) attributable to CVS Health | $ (0.57) | $ 6.47 |",
        ]
    )
    lines = table_text.splitlines()
    compact_header = parser._build_compact_repeated_table_header(lines[0], lines[2])[0]
    max_chars = len("\n".join([compact_header, lines[3], lines[4], lines[5]])) + 5

    assert len("\n".join(lines[:6])) > max_chars
    assert len("\n".join([compact_header, lines[3], lines[4], lines[5]])) <= max_chars

    parts = parser._split_table_item(
        {"text": table_text, "label": "table"},
        max_chars=max_chars,
    )

    assert len(parts) >= 2
    part_lines = [
        [line for line in part["text"].splitlines() if line.strip()]
        for part in parts
    ]

    context_part = next(
        lines for lines in part_lines if "| Per common share data: | | |" in lines
    )

    assert "| Basic earnings (loss) per common share: | | |" in context_part
    assert (
        "| Income (loss) from continuing operations attributable to CVS Health | "
        "$ (0.57) | $ 6.48 |"
    ) in context_part
