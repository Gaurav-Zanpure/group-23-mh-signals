import re
import json
import pandas as pd
import argparse
from collections import Counter

def any_re_loose(patterns):
    return re.compile(r"(?i)(" + r"|".join(patterns) + r")")

crit_terms = any_re_loose([
    r"\bsi\b", r"suicide", r"suicidal", r"kill myself", r"kill\s*my\s*self", r"\bkms\b",
    r"end it all", r"end my life", r"self[-\s]*harm", r"cutting", r"cut myself", r"bleeding",
    r"\bod\b", r"overdose", r"unalive", r"i don['’]t want to live", r"i want to die",
    r"wish i was dead", r"jump off", r"no reason to live", r"take my life", r"hurting myself"
])

mal_terms = any_re_loose([
    r"relapse", r"drunk|booze|alcohol|vodka|whiskey|blackout",
    r"weed|cannabis|marijuana|stoned|high as", r"vape|nicotine",
    r"\bcoke\b|cocaine|\bmeth\b|speed|xanax|benzo|opioid|heroin",
    r"pills|adderall|oxy|fentanyl",
    r"binge(ing)?|purge(ing)?|starv(ing|e)|restrict(ing)?",
    r"self[-\s]*medicat", r"doomscroll(ing)?", r"gambl(ing)?"
])

help_terms = any_re_loose([
    r"\bhelp\b", r"any advice", r"what should i do", r"how do i", r"tips?\b", r"resources?\b",
    r"recommend(ed|ation|ations)?\b", r"where can i", r"pls help|please help",
    r"suggestions?\b", r"hotline", r"therapist near", r"looking for a therapist",
    r"how to cope", r"need guidance", r"can someone", r"does anyone know", r"anyone else\?"
])

treat_terms = any_re_loose([
    r"\bssri\b|prozac|sertraline|zoloft|fluoxetine|lexapro|escitalopram|wellbutrin|bupropion|venlafaxine|effexor",
    r"\bmeds?\b|medication|\bdose\b|side effects?|taper(ing)?|withdrawal",
    r"psychiatrist|psychologist|therap(y|ist)|session|appt|appointment|refill",
    r"\bcbt\b|\bdbt\b|emdr|mindfulness-based", r"diagnos(e|ed|is)",
    r"group therap(y|ies)|support group", r"mental health awareness|\bawareness\b"
])

pos_terms = any_re_loose([
    r"cop(e|ing)", r"journaling|journaled", r"breath(ing)? exercises?", r"breathwork",
    r"mindful(ness)?|meditat(e|ion)", r"gratitude|affirmations?", r"grounding",
    r"recovery|resilien(ce|t)|therapy is helping|worked for me|progress|improv(ing|ed)|small wins?",
    r"stay strong|sending love|proud of you|you got this|i'm proud of", r"sober \d+ (days|weeks|months)",
    r"exercise|workout|gym|walk(ed|ing)|run(ning)?|yoga|art|music|hobbies?",
    r"humou?r|meme|lol|haha|lmao|😂|🤣|😄|😊|🙂"
])

mood_terms = any_re_loose([
    r"mood check(-|\s)?in|check(-|\s)?in", r"today i feel", r"\bi feel\b|\bfeeling\b",
    r"status:\s*\w+", r"current mood", r"how i[’']?m feeling|how im feeling",
    r"\b\d+/\d+\b|rate my day|on a scale of"
])

cause_terms = any_re_loose([
    r"break ?up|broke up|\bex\b|toxic|argu(ing|ment)", r"fight with (my )?(mom|dad|parents|friend|gf|bf|partner)",
    r"cheat(ed|ing)|ghost(ed|ing)|lonely|no friends|ignored|bully(ied|ing)|isolation|alone|left out",
    r"exam(s)?|finals|midterms?|deadline|assignment|grades?|gpa",
    r"job stress|boss|manager|layoff|interview|recruiter|offer|on[-\s]?site",
    r"rent|evict|landlord|lease|deposit|housing", r"broke\b|money|debt|loan|tuition|bills|financial",
    r"visa|immigration|sevis|opt|ead", r"moving|new city|roommate",
    r"workload|burnout|all[-\s]?nighter|sleep deprived",
    r"traffic|commute|car broke|ticket|accident|parking"
])

distress_terms = any_re_loose([
    r"depress(ed|ion)|anxious|anxiety|panic|attack|overwhelmed|overwhelm", r"stressed|stress",
    r"sad|sadness|blue|down bad", r"cry(ing)?|tears|sobb(ing)?", r"angry|anger|mad|furious|rage|frustrat(ed|ion)",
    r"hopeless|despair|doom|nihil(ism|istic)", r"worthless|useless|failure|loser",
    r"tired of life|drained|exhausted|burned out|numb|empty|can[’']?t go on|cant go on|why am i like this"
])

emoji_distress = any_re_loose([r"😞|😔|😣|😖|😫|😩|😭|😢|💔|😡|😤|☹️"])
first_person = re.compile(r"(?i)\b(i|i'm|im|i am|i’ve|i have|i feel|i feel like|my)\b")
neg_adj = any_re_loose([
    r"bad|awful|terrible|horrible|miserable|not okay|not ok|not fine|not great|struggling|rough|hard|difficult|chaotic|mess"
])

PRIORITY = [
    ("Critical Risk", crit_terms),
    ("Maladaptive Coping", mal_terms),
    ("Seeking Help", help_terms),
    ("Progress Update", treat_terms),
    ("Positive Coping", pos_terms),
    ("Mood Tracking", mood_terms),
    ("Cause of Distress", cause_terms),
    ("Mental Distress", distress_terms),
]

def tag_post_v2(text: str) -> str:
    t = (text or "").strip()
    tl = t.lower()
    for tag, pat in PRIORITY:
        if pat.search(tl):
            return tag
    if emoji_distress.search(t):
        return "Mental Distress"
    if first_person.search(tl) and neg_adj.search(tl):
        return "Mental Distress"
    if re.search(r"\?\s*$", t) and re.search(r"(?i)\b(anyone|someone|advice|how|what|where|help)\b", tl):
        return "Seeking Help"
    if first_person.search(tl) and re.search(r"(?i)\bfeel(ing)?\b", tl):
        return "Mood Tracking"
    return "Miscellaneous"

def main(input_path, output_path):
    print(f"\nReading: {input_path}")
    df = pd.read_csv(input_path)
    df["Text"] = df["Text"].fillna("").astype(str)

    df["Final_Intent_Tag"] = df["Text"].map(tag_post_v2)
    df.to_csv(output_path, index=False, encoding="utf-8")

    counts = Counter(df["Final_Intent_Tag"])
    print("\nTagging complete. Summary:\n")
    for tag, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"{tag:25s} {count}")
    print(f"\nSaved tagged data to: {output_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rule-based intent tagging for mental health dataset")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save tagged CSV")
    args = parser.parse_args()

    main(args.input_path, args.output_path)
