"""Cost comparison: steered Tiny Aya on a rented T4 vs closed-model APIs.

Timing and token counts come from the final run (final_outputs.jsonl, steered build at
strength 0.25, batch size 1, greedy, NF4 on a Colab T4). Prices come from the pages below,
checked 2026-11-08. The monthly volume is an assumption, not a measurement.

Run: python3 roi.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent

# https://huggingface.co/docs/inference-endpoints/pricing : nvidia-t4 x1, $0.5/hour, billed by the minute
T4_PER_HOUR = 0.50
HOURS_PER_MONTH = 730

# https://platform.claude.com/docs/en/about-claude/pricing : base input / output, USD per million tokens
CLOSED = {
    "Claude Haiku 5.5": (0.10, 0.50),
    "Claude Sonnet 5.5": (2.00, 10.00),
    "Claude Opus 5.5": (4.00, 20.00),
}

# Assumption: 200 electricians, 5 questions each per working day, 22 working days a month.
MONTHLY_QUESTIONS = 200 * 5 * 22


def measured(build="4bit_steer"):
    rows = [json.loads(l) for l in open(HERE / "final_outputs.jsonl")]
    rows = [r for r in rows if r["build"] == build]
    secs = sum(r["seconds"] for r in rows)
    out_tok = sum(r["new_tokens"] for r in rows)
    out_words = sum(len(r["response"].split()) for r in rows)
    in_words = sum(len(r["prompt"].split()) for r in rows)
    n = len(rows)
    # Prompt tokens are estimated from prompt words, using the tokens-per-word ratio
    # measured on this build's own answers (chat-template tokens are not counted).
    return {
        "n": n,
        "sec": secs / n,
        "out_tok": out_tok / n,
        "in_tok": in_words / n * out_tok / out_words,
        "tok_s": out_tok / secs,
    }


def main():
    base, s = measured("4bit_base"), measured("4bit_steer")
    print(f"Measured, {s['n']} answers per build, batch size 1 on a T4:")
    print(f"  base:    {base['sec']:.2f} s/answer, {base['out_tok']:.1f} output tokens, {base['tok_s']:.2f} tokens/s")
    print(f"  steered: {s['sec']:.2f} s/answer, {s['out_tok']:.1f} output tokens, {s['tok_s']:.2f} tokens/s")
    print(f"  steering throughput change: {100 * (s['tok_s'] / base['tok_s'] - 1):+.0f}%")
    print(f"  estimated prompt tokens: {s['in_tok']:.0f}\n")

    per_answer_t4 = s["sec"] / 3600 * T4_PER_HOUR
    capacity = HOURS_PER_MONTH * 3600 / s["sec"]
    busy_hours = MONTHLY_QUESTIONS * s["sec"] / 3600
    print(f"Volume (assumed): {MONTHLY_QUESTIONS:,} questions/month")
    print(f"One T4 at batch size 1 can answer {capacity:,.0f}/month if it never idles\n")

    print("| Option | Cost per answer | Monthly cost |")
    print("|---|---|---|")
    print(f"| Steered Tiny Aya, T4 endpoint, billed only while generating | ${per_answer_t4:.5f} | "
          f"${busy_hours * T4_PER_HOUR:,.2f} ({busy_hours:.0f} GPU-hours) |")
    print(f"| Steered Tiny Aya, T4 endpoint always on | n/a | ${HOURS_PER_MONTH * T4_PER_HOUR:,.2f} |")
    for name, (p_in, p_out) in CLOSED.items():
        c = (s["in_tok"] * p_in + s["out_tok"] * p_out) / 1e6
        print(f"| {name} API | ${c:.5f} | ${c * MONTHLY_QUESTIONS:,.2f} |")

    print("\nAlways-on T4 break-even (answers/month), against each API:")
    for name, (p_in, p_out) in CLOSED.items():
        c = (s["in_tok"] * p_in + s["out_tok"] * p_out) / 1e6
        be = HOURS_PER_MONTH * T4_PER_HOUR / c
        note = "within one T4's capacity" if be <= capacity else "beyond one T4's capacity"
        print(f"  {name}: {be:,.0f} ({note})")


if __name__ == "__main__":
    main()
