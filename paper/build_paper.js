const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType, LevelFormat,
  Footer, PageNumber, VerticalAlign,
} = require("docx");

const FIG = path.join(__dirname, "figures");
const CW = 9360; // content width (US Letter, 1" margins)

// ---------- helpers ----------
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
function P(runs, opts = {}) {
  const children = (Array.isArray(runs) ? runs : [runs]).map((r) =>
    typeof r === "string" ? new TextRun(r) : r);
  return new Paragraph({ children, spacing: { after: 120, line: 276 },
    alignment: opts.align || AlignmentType.JUSTIFIED, ...opts });
}
const B = (t) => new TextRun({ text: t, bold: true });
const I = (t) => new TextRun({ text: t, italics: true });
function bullet(runs) {
  const children = (Array.isArray(runs) ? runs : [runs]).map((r) =>
    typeof r === "string" ? new TextRun(r) : r);
  return new Paragraph({ numbering: { reference: "b", level: 0 },
    spacing: { after: 60 }, children });
}
function caption(t) {
  return new Paragraph({ spacing: { before: 80, after: 200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: t, italics: true, size: 19 })] });
}
function img(file, w, h) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(FIG, file)),
      transformation: { width: w, height: h },
      altText: { title: file, description: file, name: file } })] });
}
const border = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
const borders = { top: border, bottom: border, left: border, right: border,
  insideHorizontal: border, insideVertical: border };
function cell(text, width, { head = false, bold = false, align = AlignmentType.LEFT } = {}) {
  return new TableCell({ borders, width: { size: width, type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    shading: head ? { fill: "2E5A88", type: ShadingType.CLEAR } : { fill: "FFFFFF", type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 110, right: 110 },
    children: [new Paragraph({ alignment: align, children: [
      new TextRun({ text: String(text), bold: head || bold, color: head ? "FFFFFF" : "000000", size: 19 }) ] })] });
}
function table(widths, rows) {
  return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: widths,
    rows: rows.map((r, ri) => new TableRow({ tableHeader: ri === 0,
      children: r.map((c, ci) => cell(c, widths[ci],
        { head: ri === 0, align: ci === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })) })) });
}

// ---------- document ----------
const children = [];

// Title block
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
  children: [new TextRun({ text: "Where Cascaded ASR Pipelines Fail on Accented Learner Speech, and What Audio-Free Adaptation Can (and Can't) Fix", bold: true, size: 34 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
  children: [new TextRun({ text: "Hamna Zahid", size: 24 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20 },
  children: [new TextRun({ text: "Department of Data Science, The Islamia University of Bahawalpur, Pakistan", italics: true, size: 20 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
  children: [new TextRun({ text: "ask.hamnazahid@gmail.com", size: 18 })] }));

// Abstract
children.push(new Paragraph({ spacing: { after: 80 }, children: [B("Abstract.")] , border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "2E5A88", space: 2 } }}));
children.push(P([
  "Automatic speech recognition (ASR) is increasingly deployed in streaming, cascaded pipelines, yet its behaviour on second-language (L2) accented speech under streaming constraints is poorly characterised. We present a two-part study on a frozen Whisper-small decoder. ",
  B("Phase A"), " is a diagnostic that separates three error sources: (i) the streaming/chunking penalty, measured by decoding the same utterances offline and in fixed-latency chunks of 320/640/1280 ms; (ii) accent-driven errors, isolated by cross-referencing ASR errors against L2-ARCTIC's manual phoneme annotations; and (iii) residual model errors on correctly-produced speech. ",
  B("Phase B"), " asks whether an entirely ", I("audio-free"), " adaptation - n-best rescoring with text-only language models, both a weak domain n-gram and a strong neural LM - can recover any of this error on the frozen decoder. On Indian-accented Svarah and the L2-ARCTIC learner corpus we find the streaming penalty is large and monotonic (WER rises up to ~5x from offline to 320 ms chunks); roughly 51% of L2-ARCTIC errors coincide with genuine mispronunciations while the remainder are model errors on correct speech; and whether audio-free rescoring helps follows a clean 2x2 in LM strength and speech type. A weak domain n-gram never yields a test-set gain: its dev improvement fails to generalize. A strong, zero-shot GPT-2 gives an 8.0% relative WER reduction on L2-ARCTIC read speech across 24 speakers (four per L1), significant under a speaker-level cluster bootstrap (p = 0.0003) and robust across ASR model sizes (tiny/base/small.en, larger on the weaker models) - but the same strong LM gives no gain on spontaneous Svarah. The error taxonomy explains the pattern: a strong LM can re-rank lexical errors when the n-best contains a better wording, but the roughly half of errors that are accent-driven (51% by phoneme ground truth) remain beyond any text model, bounding the gain and eliminating it on spontaneous speech. Audio-free adaptation is thus a cheap, safe, modest add-on in strong-LM read-speech settings, not a general remedy for the accent gap. All code and derived results are released for reproducibility.",
]));

children.push(P([B("Keywords: "), new TextRun("speech recognition, L2/accented speech, streaming ASR, error analysis, language model rescoring, domain adaptation, Whisper.")]));

// 1. Introduction
children.push(H1("1  Introduction"));
children.push(P("Second-language (L2) English speakers form a large fraction of global ASR users, yet most ASR systems are trained and tuned on native, read, full-utterance speech. Two deployment realities make accented L2 recognition harder still. First, interactive applications run ASR in a streaming, cascaded pipeline - voice-activity detection (VAD) followed by incremental decoding - so the recogniser must emit words before the utterance is complete, losing the future acoustic context that offline decoding enjoys. Second, adapting large ASR models to a target domain or accent is expensive: it typically requires transcribed in-domain audio and gradient updates to the acoustic model, which many practitioners cannot afford."));
children.push(P("This paper studies both problems on a single frozen Whisper-small.en decoder, and asks a deliberately practical question: how much of the error on accented learner speech is caused by streaming versus accent versus the model itself, and how much can be recovered without any audio-side training at all? We make the following contributions:"));
children.push(bullet([B("A streaming-mismatch diagnostic"), new TextRun(" that quantifies the WER/latency trade-off of fixed-latency chunked decoding against an offline reference, on two accented L2 corpora.")]));
children.push(bullet([B("An accent-vs-model error decomposition"), new TextRun(" that cross-references automatically aligned ASR errors with L2-ARCTIC's ground-truth phoneme mispronunciation annotations, separating genuine accent errors from model errors on correctly-produced speech.")]));
children.push(bullet([B("An audio-free adaptation study"), new TextRun(" in which a domain n-gram language model, trained only on text and applied via n-best rescoring on the frozen decoder, is evaluated through the same error taxonomy - including an honest per-first-language fairness analysis.")]));
children.push(bullet([B("A reproducible, resource-constrained pipeline"), new TextRun(" that runs on an 8 GB CPU (with optional GPU), released as open code.")]));
children.push(P("Our central finding is a contrast: audio-free rescoring helps on constrained read speech (L2-ARCTIC) but not on open-domain spontaneous speech (Svarah), and the diagnostic explains why - the language model can only fix lexical errors, not the acoustic, accent-driven errors that dominate the harder corpus."));

// 2. Related Work
children.push(H1("2  Related Work"));
children.push(P([B("Accented and dialectal ASR disparities. "), new TextRun("A growing body of work documents that ASR underserves non-standard speakers: Koenecke et al. [5] measured large racial gaps across five commercial systems, and Tatman [6] found dialect and gender bias in automatic captions. Accent benchmarks such as Svarah [2] quantify degradation on Indian-accented English. Whisper [1], trained on web-scale weak supervision, narrows but does not close these gaps. Rather than propose a new model, we characterise where and why a frozen Whisper pipeline fails on L2 speech, and what a cheap text-only fix can recover.")]));
children.push(P([B("End-to-end and streaming ASR. "), new TextRun("Modern recognisers build on CTC [7], attention sequence-to-sequence models [8], and the Conformer [9]; Whisper [1] is an attention encoder-decoder, and self-supervised encoders [11, 12] offer an alternative acoustic-side adaptation route. Streaming systems bound latency by limiting right context [10]. We do not train a streaming model; we emulate fixed-latency chunked decoding on a frozen offline model to isolate the chunking penalty as a controlled variable.")]));
children.push(P([B("Pronunciation modelling and learner corpora. "), new TextRun("Goodness-of-pronunciation scoring [13] and neural mispronunciation detection and diagnosis [14] target L2 pronunciation directly; L2-ARCTIC [3], built on the CMU ARCTIC prompts [4], provides manual phoneme-level annotations widely used for that task. We instead repurpose these annotations as ground truth to separate accent-driven ASR errors from model errors - a diagnostic use rather than detection.")]));
children.push(P([B("External LMs and rescoring. "), new TextRun("Shallow fusion [15, 16] and n-best rescoring add an external LM to a frozen decoder; KenLM [17] enables efficient n-gram scoring, while large neural LMs such as GPT-2 [18], built on the Transformer [19], provide much stronger text priors. We compare a weak n-gram and a strong neural LM as an audio-free adaptation, and - unusually - evaluate the outcome through an error taxonomy and a paired bootstrap significance test [20, 21] rather than by WER alone. A recent line of work instead has an LLM generate a corrected transcript from the n-best (generative error correction): HyPoradise [25], Whispering-LLaMA [26] and the GenSEC challenge [27] show such models can recover words absent from the n-best, surpassing the re-ranking upper bound. We deliberately study the cheaper, fully reproducible re-ranking regime - no generation, runs on a free GPU - and quantify where its oracle ceiling lies and why a strong LM stops short of it.")]));

// 3. Method
children.push(H1("3  Method"));
children.push(H2("3.1  Cascaded pipeline"));
children.push(P("Figure 1 shows the pipeline. Audio is segmented with Silero VAD [22] and decoded by a frozen Whisper-small.en model via faster-whisper/CTranslate2 [23]. The same frozen decoder feeds both studies: Phase A compares offline and streaming decoding and builds an error taxonomy; Phase B extracts scored n-best hypotheses for text-only LM rescoring."));
children.push(img("fig1_pipeline.png", 560, 230));
children.push(caption("Figure 1: The cascaded ASR pipeline and the two studies. The Whisper decoder is frozen throughout; Phase B adds only a text-trained language model at the n-best stage."));

children.push(H2("3.2  Datasets"));
children.push(P("We evaluate on two accented L2 English corpora (Table 1). Svarah [2] is Indian-accented English spanning many first languages (L1s) and both read and spontaneous speech. L2-ARCTIC [3] is read speech from learners of six L1s with manual phoneme annotations. Given a deliberately resource-constrained setting, we evaluate on seeded subsamples. The Phase B headline adaptation result uses all 24 L2-ARCTIC speakers (four per L1, 1200 clips); the Phase A diagnostic and the cross-model-size comparison (Table 5) use a fixed six-speaker subset (one per L1), so that the phoneme-aligned error analysis and the model-size differences respectively are not confounded by speaker mix. All audio is resampled to 16 kHz mono."));
children.push(table([2400, 2400, 2280, 2280], [
  ["Property", "Svarah", "L2-ARCTIC", "Role"],
  ["Accent / L1s", "19 Indian L1s", "6 L1s", "coverage"],
  ["Style", "read + spontaneous", "read prompts", "domain"],
  ["Phoneme labels", "no", "yes (manual)", "accent ground truth"],
  ["Phase A clips", "150", "300 (6 spk)", "diagnostic"],
  ["Phase B eval clips", "150", "1200 (24 spk, 4/L1)", "adaptation"],
]));
children.push(caption("Table 1: Evaluation corpora. Subsamples are seeded and reproducible; clips shorter than 1 s (single-word commands) are excluded so streaming chunk effects are well-defined."));

children.push(H2("3.3  Phase A: streaming and error diagnostics"));
children.push(P([new TextRun("Each clip is decoded under four conditions: offline (full utterance) and streaming with fixed chunks of 1280, 640 and 320 ms. Streaming is emulated by feeding each chunk a bounded left-context window but "), I("no"), new TextRun(" future audio, isolating the loss of right context that characterises real streaming systems; the algorithmic latency equals the chunk size. We report WER and CER (via jiwer [24]) and real-time factor.")]));
children.push(P("For the error taxonomy, every hypothesis is aligned to its reference and each substitution/deletion/insertion is logged with full reference and hypothesis sentences, local context, and an approximate timestamp. Each error is then assigned a cause by a transparent, reproducible procedure that uses the strongest signal available: (i) for L2-ARCTIC, errors overlapping an annotated mispronunciation (via the phoneme join) are labelled accent-driven; (ii) orthographic-only differences (British/American spelling, digit-vs-word number forms, contractions) are normalization artifacts; (iii) when the reference text is a substring of the hypothesis (a truncated reference), the surplus insertions are reference errors and the model was effectively correct; (iv) repeated inserted words are disfluencies and unexplained insertions are hallucinations; remaining substitutions are model-lexical errors. Accent is asserted only from phoneme ground truth: we tested a text-only phonetic-similarity proxy (metaphone / Jaro-Winkler) for use where phoneme labels are absent (notably Svarah), but validated against the L2-ARCTIC ground truth it was uncorrelated with the true labels (agreement 49%, Cohen's kappa ~ 0), so we do not use it. This is an analysis aid, not a substitute for human listening (see Limitations)."));

children.push(H2("3.4  Phase B: audio-free n-best rescoring"));
children.push(P([new TextRun("Phase B never touches the acoustic model. For each clip we extract the decoder's top-"), I("N"), new TextRun(" hypotheses with their acoustic scores (via CTranslate2's num_hypotheses / return_scores), then re-rank them as")]));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 120 },
  children: [new TextRun({ text: "score(h) = s_acoustic(h) + λ · logP_LM(h) / |h|,", italics: true })] }));
children.push(P([new TextRun("where the LM term is the per-word log-probability under a domain n-gram model and λ is a fusion weight. The LM is a 4-gram trained "), I("only on text"), new TextRun(". To prevent leakage - critical because all L2-ARCTIC speakers read the same prompts - every sentence matching a test reference is removed from the LM corpus. λ is tuned on a held-out dev split and the WER delta is reported on a disjoint test split, both overall and stratified by L1. Choosing λ = 0 recovers the unadapted top-1; but because λ is selected on dev, a positive weight can still raise WER on the disjoint test split - precisely what separates the weak from the strong LM below.")]));

// 4. Setup
children.push(H1("4  Experimental Setup"));
children.push(P("The backbone is Whisper-small.en (244 M parameters) in faster-whisper/CTranslate2, int8 on CPU and float16 on GPU; beam size 5 (offline) and 10 for n-best (N = 10). We rescore with two LMs of contrasting strength: a weak, pure-Python interpolated 4-gram trained on the in-domain text corpus, and a strong, zero-shot GPT-2 (124 M) scoring each hypothesis by its token log-likelihood. Because the n-best is decoded once and cached, both rescorers run cheaply on the same hypotheses. Phase A on Svarah ran on an 8 GB-RAM CPU (Intel i5-4300U); the heavier all-condition and n-best decoding ran on a free Colab T4 GPU. Offline WER was identical across CPU (int8) and GPU (float16) decoding (0.132 on Svarah), confirming the hardware swap does not affect results. All scores use jiwer [24] with light text normalization (lower-case, punctuation removal)."));

// 5. Results
children.push(H1("5  Results"));
children.push(H2("5.1  The streaming penalty"));
children.push(P("Table 2 and Figure 2 show that WER rises sharply and monotonically as the chunk shrinks. Relative to offline, 320 ms streaming is 4.8x worse on L2-ARCTIC and 5.3x worse on Svarah. The degradation is driven by the number of chunks, not context length: because the encoder pads every chunk to its full window, smaller chunks impose far more recomputation and far less right context."));
children.push(table([2160, 1800, 1800, 1800, 1800], [
  ["Condition", "Svarah WER", "Svarah CER", "L2-ARCTIC WER", "L2-ARCTIC CER"],
  ["Offline", "0.132", "0.081", "0.101", "0.053"],
  ["Stream 1280 ms", "0.364", "0.211", "0.199", "0.121"],
  ["Stream 640 ms", "0.477", "0.291", "0.304", "0.196"],
  ["Stream 320 ms", "0.705", "0.451", "0.480", "0.310"],
]));
children.push(caption("Table 2: Mean WER/CER by decoding condition. Streaming degrades accuracy monotonically; the 320 ms condition is roughly 5x the offline WER on both corpora."));
children.push(img("fig2_latency_wer.png", 420, 282));
children.push(caption("Figure 2: Streaming latency vs WER. Dashed lines are the offline baselines; markers are streaming conditions (x-axis inverted so latency decreases rightward)."));
children.push(P("Figure 3 shows the failure qualitatively on a real clip: at 320 ms, with no future context, the frozen decoder fragments the sentence and even hallucinates fluent filler, turning a perfect offline transcript into WER > 1."));
children.push(img("fig_streaming_break.png", 452, 336));
children.push(caption("Figure 3: Streaming breakage on a real L2-ARCTIC clip that offline transcribes perfectly. The mel-spectrogram is overlaid with 320 ms chunk boundaries (dashed) and the offline word alignment; decoded chunk-by-chunk with no future context, the model fragments and hallucinates (\"I will see you in the next video\", \"courgette\"), WER 0.00 -> 1.18."));

children.push(H2("5.2  Accent vs. model errors"));
children.push(P("Using L2-ARCTIC's phoneme ground truth, 136 of 264 aligned errors (51.5%) coincide directly with an annotated mispronunciation; the remaining 48.5% fall on correctly-produced speech. The coincidence varies by operation: 56% of substitutions and 60% of deletions are accent-aligned, but only 29% of insertions - consistent with insertions being hallucinations rather than accent effects. The fine taxonomy (Table 3, Figure 4) labels 45.5% of L2-ARCTIC errors as accent (phoneme-confirmed; some coinciding errors are better explained as normalization or reference errors, which take precedence). We deliberately report no accent share for Svarah: lacking phoneme labels one might substitute a text-only phonetic-similarity proxy (metaphone / Jaro-Winkler), but validated against the L2-ARCTIC ground truth this proxy was uncorrelated with the true accent labels (agreement 49%, Cohen's kappa ~ 0), so we do not use it. Svarah is characterised only by categories text rules determine reliably (model-lexical, hallucination, normalization, deletion)."));
children.push(table([2760, 3300, 3300], [
  ["Error category", "L2-ARCTIC (% of 264)", "Svarah (% of 160)"],
  ["Accent (phoneme-confirmed)", "45.5", "— (no labels)"],
  ["Model lexical", "29.5", "55.0"],
  ["Hallucination (insertion)", "11.4", "21.3"],
  ["Normalization artifact", "9.1", "7.5"],
  ["Deletion (other)", "2.3", "14.4"],
  ["Reference error", "2.3", "1.9"],
]));
children.push(caption("Table 3: Error-category composition (offline). L2-ARCTIC accent labels use human phoneme ground truth; Svarah has none, and a text-only phonetic proxy was found uncorrelated with that ground truth (Cohen's kappa ~ 0), so no accent share is reported for Svarah."));
children.push(img("fig3_error_categories.png", 430, 264));
children.push(caption("Figure 4: Error-category composition for both corpora. Only L2-ARCTIC supports a phoneme-confirmed accent category."));

children.push(H2("5.3  Audio-free adaptation"));
children.push(P("Whether audio-free adaptation helps depends on two factors, which together form a clean 2x2 (Table 4, Figure 5): the strength of the LM, and whether the speech is read or spontaneous. We rescore the same cached n-best with a weak domain 4-gram and a strong, zero-shot GPT-2."));
children.push(P("The weak n-gram never yields a test-set gain. On the six-speaker subset and on Svarah its dev optimum is λ = 0 - it cannot out-rank Whisper's own internal LM at all - while on the full 24-speaker set the small dev improvement it finds at λ = 0.3 fails to transfer, raising test WER by 0.8% (Figure 5, blue star above baseline): a small-corpus n-gram overfits the dev split. The strong neural LM behaves very differently. On L2-ARCTIC read speech - evaluated on all 24 speakers (four per L1, 1200 clips), which removes the single-speaker-per-L1 confound - its dev gain transfers to test (Figure 5, red star below baseline), reducing test WER from 0.084 to 0.077, an 8.0% relative reduction that improves five of the six L1s. We assess significance with a paired bootstrap. The improvement is significant at the utterance level (tail p < 0.001) and, crucially, under a speaker-level cluster bootstrap that resamples the 24 speakers to account for within-speaker correlation (p = 0.0003); the speaker-level 95% CI on the absolute delta, [-0.011, -0.003], excludes zero (two-sided significant at α = 0.05), so the effect is not an artifact of any single speaker. It is in line with the ~9% relative gains reported for shallow fusion on attention seq2seq models [16]. The benefit also holds across ASR model size (Table 5, Figure 6): the GPT-2 gain is significant for tiny.en, base.en and small.en, and is larger on the weaker models, which make more lexical errors for the LM to repair. To compare sizes cleanly we hold the speaker set fixed at the six-speaker subset (198 test utterances) for all three models; on this controlled subset small.en gives -6.4% (p = 0.030), consistent with - and more conservative than - the -8.0% it reaches on the full 24 speakers (Table 4). A category audit attributes the fixes to lexical error (model-lexical substitutions and number-form normalization), consistent with Table 6. On spontaneous Svarah, even GPT-2 yields no improvement (Δ = +0.001, p = 0.74)."));
children.push(table([2400, 3480, 3480], [
  ["WER delta (test)", "4-gram LM (weak)", "GPT-2 LM (strong)"],
  ["L2-ARCTIC (read, 24 spk)", "+0.001, +0.8%  (n.s.)", "-0.0068, -8.0%  (p < 0.001)"],
  ["Svarah (spontaneous)", "0.000  (λ=0, n.s.)", "+0.0009, +0.9%  (p = 0.74)"],
]));
children.push(caption("Table 4: Audio-free n-best rescoring - WER change by LM strength and speech type (test split). Negative = improvement. Adaptation helps only in one cell: a strong LM on read speech. A weak LM yields no test gain; even a strong LM fails on spontaneous speech."));
children.push(img("fig4_phaseB.png", 430, 287));
children.push(caption("Figure 5: Dev-set WER versus fusion weight λ on L2-ARCTIC (24 speakers); stars mark the realized test outcome at each LM's dev-selected λ. Both LMs lower dev WER, but only the strong GPT-2's gain generalizes (test -8.0%, star below baseline); the weak 4-gram's dev-optimal λ raises test WER (+0.8%, star above baseline). LM strength decides whether audio-free adaptation transfers."));
children.push(table([2520, 1560, 1560, 1560, 2160], [
  ["ASR model", "Base WER", "Adapt WER", "delta rel", "p"],
  ["tiny.en (39M)", "0.147", "0.133", "-9.5%", "0.010"],
  ["base.en (74M)", "0.124", "0.109", "-11.8%", "0.005"],
  ["small.en (244M)", "0.088", "0.082", "-6.4%", "0.030"],
]));
children.push(caption("Table 5: Strong-LM (GPT-2) gain across ASR model size, on a fixed six-speaker L2-ARCTIC subset (198 test utterances) held constant so that differences reflect model size, not speaker mix. Significant for every model size and larger on the weaker models. On this controlled subset small.en gives -6.4%, versus the -8.0% headline when small.en is run on all 24 speakers (Table 4)."));
children.push(img("fig5_model_robustness.png", 380, 266));
children.push(caption("Figure 6: Relative WER reduction from GPT-2 rescoring across ASR model size - significant for all, larger on weaker models."));
children.push(P("Figure 7 traces one such fix end-to-end: on a real clip the decoder's top acoustic hypothesis mis-hears the homophone \"were\" as \"where\", and the correct wording - ranked only sixth by acoustics - is promoted to first by the LM."));
children.push(img("fig_hero_rescore.png", 430, 369));
children.push(caption("Figure 7: How a text LM repairs accented ASR, on a real L2-ARCTIC clip (top: mel-spectrogram). Whisper's acoustic n-best ranks the homophone error \"WHERE\" first; the grammatical wording \"were\" sits sixth by acoustic score but first under acoustic + lambda*LM, so rescoring promotes it (green arrow). Utterance WER 0.08 -> 0.00. A sound the acoustics cannot disambiguate, fixed by the language model."));
children.push(P("Table 6 shows representative GPT-2 corrections on L2-ARCTIC: number forms, rare-word recovery, homophone disambiguation from context, and grammatical agreement - the lexical errors a strong LM can identify and the decoder's n-best happens to contain."));
children.push(table([3000, 3180, 3180], [
  ["Reference (fragment)", "Baseline hypothesis", "Adapted hypothesis"],
  ["three hundred yards apart", "300 yards apart", "three hundred yards apart"],
  ["over the handkerchief", "over the hand crochet", "over the handkerchief"],
  ["the fourth and fifth days", "the 4th and 5th days", "the fourth and fifth days"],
  ["the gray eyes faltered", "the gray ice faltered", "the gray eyes faltered"],
  ["there would have to be", "there will have to be", "there would have to be"],
]));
children.push(caption("Table 6: Representative L2-ARCTIC corrections from strong-LM (GPT-2) rescoring. Each lowers utterance WER to zero or near-zero."));

children.push(H2("5.4  How much error is recoverable? An oracle ceiling and an LM-strength ladder"));
children.push(P("How much of the residual error can any audio-free rescorer reach? We answer with two analyses on the same 24-speaker test split, both purely on the cached n-best (no audio)."));
children.push(P([B("Oracle ceiling. "), new TextRun("If an oracle always picked the lowest-WER hypothesis in the top-10, test WER would fall from 0.084 to 0.042 - half the error is already present, correctly transcribed, somewhere in the n-best (Figure 8). The recoverable fraction grows monotonically with depth (WER 0.073 / 0.062 / 0.053 / 0.042 at N = 2 / 3 / 5 / 10) and has not saturated at ten. The bottleneck is therefore not that the decoder lacks the right words, but that selecting them needs information the top-1 acoustic ranking lacks.")]));
children.push(P([B("LM-strength ladder. "), new TextRun("Does a stronger LM climb toward this ceiling? We rescore the identical n-best with a ladder of LMs from a 4-gram to GPT-2-xl, taking each LM's word-level perplexity on the held-out references as its strength (Table 7, Figure 9). The gain rises smoothly as the LM strengthens - +0.8% (the 4-gram hurts), -3.3%, -8.0%, -11.9% - and then saturates: GPT-2-medium (355M), large (774M) and xl (1.5B) are essentially tied at about -12%, so a 4x increase in LM size past medium buys nothing. The plateau captures only about a quarter of the oracle-recoverable head-room; the other ~76% sits in the n-best but is invisible to text, because the cue that distinguishes the correct hypothesis is acoustic, not lexical. This is the quantitative form of the accent bound: the ceiling on audio-free adaptation is set by acoustics, not by LM capacity. (The significance-tested -8.0% of Section 5.3 corresponds to the GPT-2-124M rung; the larger LMs only deepen the effect.)")]));
children.push(P([B("Fine-tuning vs. scale. "), new TextRun("A GPT-2 (124M) fine-tuned on the leakage-guarded in-domain corpus - whose training text shares no sentence with any test prompt - reaches -11.4%, matching GPT-2-medium, a model 3x larger, and the plateau (Table 7). Genuine, audio-free domain adaptation thus delivers what scale does at a fraction of the cost, but neither breaks the acoustic ceiling.")]));
children.push(table([3000, 1300, 1300, 1560, 1560], [
  ["LM", "ppl", "lambda*", "delta rel", "head-room"],
  ["4-gram (weak)", "1017", "0.3", "+0.8%", "-2%"],
  ["distilGPT-2 (82M)", "134", "0.2", "-3.3%", "7%"],
  ["GPT-2 (124M)", "67", "0.2", "-8.0%", "16%"],
  ["GPT-2-medium (355M)", "54", "0.3", "-11.9%", "24%"],
  ["GPT-2-large (774M)", "49", "0.2", "-11.9%", "24%"],
  ["GPT-2-xl (1.5B)", "47", "0.2", "-11.5%", "23%"],
  ["GPT-2 fine-tuned (124M)", "54", "0.3", "-11.4%", "23%"],
]));
children.push(caption("Table 7: LM-strength ladder on L2-ARCTIC (24 speakers, test split). Strength = word-level perplexity on held-out references (lower is stronger). The gain scales with strength then saturates by GPT-2-medium; a fine-tuned 124M model matches a 3x-larger general model. Head-room % is the fraction of the oracle-recoverable error (baseline -> 0.042) captured."));
children.push(img("fig_oracle_headroom.png", 470, 292));
children.push(caption("Figure 8: Half the error is recoverable from the n-best: an oracle picking the best of the top-N hypotheses reaches WER 0.042 (from 0.084), and recoverability keeps growing with depth. A strong text LM (GPT-2) captures only ~16% of this head-room - the rest is acoustic/accent error that sits in the list but is invisible to a language model."));
children.push(img("fig_lm_ladder.png", 480, 303));
children.push(caption("Figure 9: Rescoring gain versus LM strength (word-level perplexity, stronger rightward). The gain scales with LM strength then saturates at about -12%, far above the -50% oracle ceiling; the shaded band is the accent/acoustic error that no text LM reaches. A fine-tuned 124M model (star) reaches the same plateau as a 1.5B general model."));

children.push(H2("5.5  Why it helps read speech but not spontaneous speech"));
children.push(P("The two phases explain the 2x2. A text-only LM can only adjudicate between hypotheses that differ in their words, and only if a better wording is actually present in the n-best. Two things must therefore align. First, the LM must be strong enough to score natural English reliably: the 4-gram, trained on a small corpus, cannot beat Whisper's internal LM and only adds noise, whereas GPT-2 carries enough world knowledge to prefer the correct wording. Second, the errors must be lexical and the better hypothesis must be in the list: on L2-ARCTIC read speech, enough errors are lexical (number forms, rare words, contextual homophones) for a strong LM to help; on spontaneous Svarah, errors are more acoustic and varied and the n-best rarely contains a better wording, so even GPT-2 cannot help. The gain is real and robust where it occurs but inherently bounded: about half of errors are accent-driven (51% by phoneme ground truth) and beyond any text model, which is why even the strong-LM benefit caps in the single-digit-to-low-teens percent range and disappears on spontaneous speech. Audio-free adaptation is thus a cheap, effective complement for strong-LM read-speech settings, not a general substitute for acoustic-side adaptation."));

// 6. Discussion
children.push(H1("6  Discussion"));
children.push(P("Three practical implications follow. First, for streaming deployment on accented speech, chunk size is a first-order accuracy lever: moving from 1280 ms to 320 ms roughly doubles WER, so latency budgets should be set with accented users in mind. Second, headline WER conflates distinct failure modes; the ~50/50 accent-vs-model split means that nearly half of 'accent errors' are actually model errors that better acoustic modelling - not pronunciation training - would fix. Third, audio-free text adaptation is a useful, cheap complement when used correctly: a strong neural LM gives a significant read-speech WER reduction that holds across 24 speakers and across ASR model sizes, but it requires a strong LM (a domain n-gram is useless), helps only where errors are lexical (read, not spontaneous speech), and is bounded by the accent/acoustic errors that dominate. It complements rather than replaces acoustic-side adaptation. Methodologically, our evaluation is a caution in both directions: one speaker per language can mislead either way - a partial three-L1 subset earlier produced a spurious 15.5% gain that vanished, while the six-speaker test understated an effect that strengthened and reached high significance only on 24 speakers - so full speaker/L1 coverage and speaker-level significance are essential."));

// 7. Limitations
children.push(H1("7  Limitations"));
children.push(bullet("Scale: results use seeded subsamples; the read-speech effect is confirmed on all 24 L2-ARCTIC speakers (four per L1, speaker-level p = 0.0003) and across three ASR model sizes, but the corpora remain modest in size and English-only, and Svarah is evaluated on 150 clips. Larger and more diverse evaluation would further tighten the estimates."));
children.push(bullet("Annotation: the error categorization is semi-automated. L2-ARCTIC accent labels rest on human phoneme annotations; we tested a text-only phonetic proxy for use where such labels are absent (Svarah) but found it uncorrelated with the ground truth (Cohen's kappa ~ 0), so we report no accent share for Svarah. The remaining labels are conservative text rules. An automated faithfulness audit finds 94.3% of the 264 labels satisfy their definitional rule, and a manual re-adjudication of a stratified 50-error sample agreed with the automatic labels in 88% (44/50) of cases. The disagreements concentrate at two boundaries the text rules handle crudely: number-format strings (\"three hundred\"/\"300\") tagged as deletion or accent rather than normalization, and word-splitting insertions (e.g. \"Anyway\" -> \"in a way\") tagged as hallucination - neither of which affects the accent-vs-model headline, which rests on the phoneme ground truth. We release the full 50-error sheet (with audio and seek times) for independent validation; labels are text-level, not audio-perceptually verified."));
children.push(bullet("LMs tested span a domain 4-gram, zero-shot GPT-2 from 82M to 1.5B, and a fine-tuned in-domain GPT-2; all re-rank the fixed n-best. We did not test first-pass shallow fusion, nor generative error correction, in which an LLM rewrites the transcript and can recover words absent from the n-best, surpassing the re-ranking oracle [25, 26, 27]. That route may break the acoustic ceiling we report, at substantially higher cost; whether it also overcomes the spontaneous-speech null is an open question."));
children.push(bullet("Orthographic normalization (e.g. British/American spelling) inflates raw WER and may slightly understate adaptation gains."));

// 8. Conclusion
children.push(H1("8  Conclusion"));
children.push(P("We presented a diagnostic-then-adaptation study of cascaded ASR on accented learner speech. Streaming imposes a large, monotonic penalty (up to ~5x WER); about half of L2-ARCTIC errors are genuine accent mispronunciations and half are model errors on correctly-produced speech; and audio-free n-best rescoring helps only when two conditions both hold - a strong LM and lexically-fixable read speech. A weak n-gram yields no test gain (its dev improvement does not transfer); a strong GPT-2 cuts L2-ARCTIC read-speech WER by 8.0% relative across 24 speakers (speaker-level p = 0.0003), a gain that holds across ASR model sizes, but does nothing for spontaneous Svarah. The error taxonomy explains the pattern: text fixes lexical errors but not the acoustic, accent-driven errors that dominate. Audio-free adaptation is therefore a cheap, effective complement in the right setting - not a general substitute for acoustic-side methods. We release the full pipeline - diagnostics, taxonomy, weak- and strong-LM rescoring, and significance testing - to support reproducible, honestly-evaluated work on equitable ASR for second-language speakers."));

// Ethics, Reproducibility, Acknowledgments
children.push(H1("Ethics Statement"));
children.push(P("This work uses only publicly released, consented speech corpora (Svarah and L2-ARCTIC) under their respective research licenses; no new human-subjects data were collected. The study is motivated by equity: ASR systematically underserves accented and second-language speakers [5, 6], and our diagnostics are intended to help close that gap rather than to profile speakers. We report a negative-leaning result honestly (a weak LM yields no usable gain; a strong LM helps only read speech, modestly) to avoid overstating cheap fixes. No audio is redistributed; only derived metrics and code are released. Speaker first-language labels are used solely for aggregate fairness analysis."));
children.push(H1("Reproducibility and Data Availability"));
children.push(P("All code, configuration, and derived results (WER/CER tables, error taxonomies, rescoring outputs, significance estimates) are released under an MIT license and archived on Zenodo (DOI: 10.5281/zenodo.XXXXXXX); a preprint is available at arXiv:XXXX.XXXXX. (The DOI and arXiv identifier are inserted on release.) No audio is included: each dataset is fetched from its official source by a provided script and converted to a common format. Random subsampling, dev/test splits, and bootstrap resampling use fixed seeds. The Phase A streaming sweep runs on an 8 GB-RAM CPU; the all-condition and n-best decoding run on a single free-tier T4 GPU. Offline WER matched across CPU (int8) and GPU (float16) decoding, confirming hardware-independence of the reported numbers."));
children.push(H1("Acknowledgments"));
children.push(P("We thank the AI4Bharat and Texas A&M PSI teams for releasing the Svarah and L2-ARCTIC corpora, and the maintainers of faster-whisper, CTranslate2, and Hugging Face Transformers."));

// References
children.push(H1("References"));
const refs = [
  "A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust Speech Recognition via Large-Scale Weak Supervision,” in Proc. ICML, 2023.",
  "T. Javed, S. Joshi, V. Nagarajan, et al., “Svarah: Evaluating English ASR Systems on Indian Accents,” in Proc. Interspeech, 2023.",
  "G. Zhao, S. Sonsaat, A. Silpachai, I. Lucic, E. Chukharev-Hudilainen, J. Levis, and R. Gutierrez-Osuna, “L2-ARCTIC: A Non-native English Speech Corpus,” in Proc. Interspeech, 2018.",
  "J. Kominek and A. W. Black, “The CMU ARCTIC Speech Databases,” in Proc. ISCA Speech Synthesis Workshop, 2004.",
  "A. Koenecke, A. Nam, E. Lake, et al., “Racial Disparities in Automated Speech Recognition,” Proc. National Academy of Sciences, vol. 117, no. 14, pp. 7684–7689, 2020.",
  "R. Tatman, “Gender and Dialect Bias in YouTube’s Automatic Captions,” in Proc. Workshop on Ethics in NLP, 2017.",
  "A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber, “Connectionist Temporal Classification,” in Proc. ICML, 2006.",
  "W. Chan, N. Jaitly, Q. Le, and O. Vinyals, “Listen, Attend and Spell,” in Proc. ICASSP, 2016.",
  "A. Gulati et al., “Conformer: Convolution-augmented Transformer for Speech Recognition,” in Proc. Interspeech, 2020.",
  "Y. He et al., “Streaming End-to-End Speech Recognition for Mobile Devices,” in Proc. ICASSP, 2019.",
  "A. Baevski, H. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations,” in Proc. NeurIPS, 2020.",
  "W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, “HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units,” IEEE/ACM TASLP, vol. 29, 2021.",
  "S. M. Witt and S. J. Young, “Phone-Level Pronunciation Scoring and Assessment for Interactive Language Learning,” Speech Communication, vol. 30, no. 2–3, pp. 95–108, 2000.",
  "W.-K. Leung, X. Liu, and H. Meng, “CNN-RNN-CTC Based End-to-End Mispronunciation Detection and Diagnosis,” in Proc. ICASSP, 2019.",
  "C. Gulcehre et al., “On Using Monolingual Corpora in Neural Machine Translation,” arXiv:1503.03535, 2015.",
  "A. Kannan, Y. Wu, P. Nguyen, T. N. Sainath, Z. Chen, and R. Prabhavalkar, “An Analysis of Incorporating an External Language Model into a Sequence-to-Sequence Model,” in Proc. ICASSP, 2018.",
  "K. Heafield, “KenLM: Faster and Smaller Language Model Queries,” in Proc. WMT, 2011.",
  "A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, “Language Models are Unsupervised Multitask Learners,” OpenAI Technical Report, 2019.",
  "A. Vaswani et al., “Attention Is All You Need,” in Proc. NeurIPS, 2017.",
  "M. Bisani and H. Ney, “Bootstrap Estimates for Confidence Intervals in ASR Performance Evaluation,” in Proc. ICASSP, 2004.",
  "P. Koehn, “Statistical Significance Tests for Machine Translation Evaluation,” in Proc. EMNLP, 2004.",
  "Silero Team, “Silero VAD: Pre-trained Enterprise-Grade Voice Activity Detector,” GitHub, 2021.",
  "SYSTRAN, “faster-whisper: Fast Inference for Whisper Using CTranslate2,” GitHub, 2023.",
  "N. Vaessen et al., “jiwer: Evaluate ASR with WER/CER,” software, 2018.",
  "C. Chen, Y. Hu, C.-H. H. Yang, S. M. Siniscalchi, P.-Y. Chen, and E. S. Chng, “HyPoradise: An Open Baseline for Generative Speech Recognition with Large Language Models,” in Proc. NeurIPS Datasets and Benchmarks Track, 2023.",
  "S. Radhakrishnan, C.-H. H. Yang, S. A. Khan, R. Kumar, N. A. Kiani, D. Gomez-Cabrero, and J. N. Tegner, “Whispering LLaMA: A Cross-Modal Generative Error Correction Framework for Speech Recognition,” in Proc. EMNLP, 2023.",
  "C.-H. H. Yang et al., “Large Language Model Based Generative Error Correction: A Challenge and Baselines for Speech Recognition, Speaker Tagging, and Emotion Recognition,” arXiv:2409.09785, 2024.",
];
refs.forEach((r, i) => children.push(new Paragraph({ spacing: { after: 60 },
  children: [new TextRun({ text: `[${i + 1}]  `, bold: true, size: 19 }), new TextRun({ text: r, size: 19 })] })));

children.push(new Paragraph({ spacing: { before: 200 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "Code and derived results: project repository (results contain no audio; datasets retain their own licenses).", italics: true, size: 18 })] }));

// ---------- assemble ----------
const doc = new Document({
  creator: "Hamna Zahid",
  title: "Where Cascaded ASR Pipelines Fail on Accented Learner Speech",
  styles: {
    default: { document: { run: { font: "Arial", size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: "1F3D63" },
        paragraph: { spacing: { before: 260, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, font: "Arial", color: "2E5A88" },
        paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
      alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 520, hanging: 260 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Page ", size: 18 }),
        new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(path.join(__dirname, "ASR_L2_paper.docx"), buf);
  console.log("wrote ASR_L2_paper.docx", buf.length, "bytes");
});
