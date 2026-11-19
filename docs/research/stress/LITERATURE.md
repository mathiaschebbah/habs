# Literature review: physiological and EEG correlates of stress reactions to short videos

We checked every reference on 2026-11-20 (PubMed, Crossref, Semantic Scholar or PMC full text). We quote numbers only from texts we read, and we name the source when a claim is second-hand.

## 1. Stress detection from physiological signals

Giannakakis et al. [1] review which biosignal features change consistently under stress. WESAD [2] (15 participants; Trier Social Stress Test, amusement, baseline) reports up to 80% accuracy for three classes and 93% for stress versus non-stress, under leave-one-subject-out (LOSO) validation according to [3]. Schmidt et al. [3] report accuracies from 40% to 95% across studies, with lower scores under LOSO than under n-fold validation. Saeb et al. [4] showed that record-wise cross-validation often massively overestimates accuracy, and that almost half of the wearable studies they reviewed used it. Our label rule follows Giannakakis et al. [5] (self-rated valence and arousal of videos, 18 subjects). **Comparison.** Our AUC of about 0.60 is far below WESAD, which contrasts minutes of social-evaluative stress with rest. A LOSO study in [3] using EEG, EDA and EMG scored below 58.5% for valence and 63.8% for arousal, which is closer to our regime.

## 2. Cardiac response to aversive stimuli

In picture viewing, cardiac deceleration is largest for threat, violent death and erotica [6]. Bradley [7] describes a prolonged deceleration in the first 2 s for every novel picture, larger for unpleasant than for pleasant or neutral pictures. She reads it as sensory intake akin to fear bradycardia. Freezing comes with parasympathetic deceleration, fight-or-flight with sympathetic acceleration [13, 14]. In film studies, Palomba et al. [10] (n = 46) found deceleration during a surgery film and sympathetic activation during a violent-threat film. Hagenaars et al. [12] (n = 50) found deceleration and reduced body sway (from 1 to 2 s after onset) only for the unpleasant film. Kreibig [8] reviewed 134 studies and found specificity once subtypes are separated. We could not read her full text; according to Siegel et al. [9], she proposed:

- heart rate rises in core disgust and falls in body-boundary-violation disgust;
- heart rate rises in activated sadness and falls in deactivated sadness.

Siegel's meta-analysis of 202 studies found heterogeneous heart rate increases for anger, fear and sadness, too few heart rate data for disgust, and no category-specific autonomic change. Kreibig et al. [11] separated fear, sadness and neutral films with 85% accuracy. **Comparison.** Our lower heart rate on stress trials (d = -0.12, and -0.17 against excitement) fits orienting and fear bradycardia, including the larger deceleration for unpleasant content [7]. Typical fear and anger inductions raise heart rate instead [9], so our effect more likely reflects early attentive intake or disgust-type content than active defence.

## 3. Skin conductance

Skin conductance tracks arousal, not valence [6, 15]. SCR increases are similar for anger, fear, disgust, happiness and sadness (d = 0.45 to 0.61), against 0.15 for neutral inductions [9]. Typical scoring windows are 1 to 3 s or 1 to 5 s after the stimulus. Benedek and Kaernbach [16] estimate an onset latency close to 1.3 s and call 1 to 4 s a conservative window, to be adapted for longer stimuli. Sjouwerman and Lonsdorf [17] largely confirmed these criteria. **Comparison.** Our SCR rise (+0.20 at 1 to 6 s) and the arousal coupling of our peripheral signals agree with this literature, and explain why peripheral signals cannot separate stress from excitement.

## 4. EEG correlates

Picture arousal modulates alpha desynchronization like the late positive potential [18]. Schubring and Schupp [19] (n = 16) found lower alpha and lower beta power 600 to 1,000 ms after high-arousal pictures, compared with low-arousal pictures; both sets were pleasant. Alpha desynchronization is also enhanced for both erotica and mutilation [40], so it indexes arousal and attention, not negative valence. Katmah et al. [22] attribute contradictory EEG stress findings to differing protocols and processing. Reznik and Allen [20] is a narrative review, not a meta-analysis. In a meta-analysis of frontal alpha asymmetry in depression, van der Vinne et al. [21] found d = -0.007 across 16 studies. Delta has been linked to motivation and salience detection [23]. Delta is also where ocular artifacts concentrate:

- blinks and eye movements affect the whole spectrum at single sites, and they distort alpha and beta mainly at frontopolar sites [24];
- ICA removes ocular artifacts reliably only when the components are identified objectively, ideally with eye tracking [25];
- blink inhibition itself varies with film content [10].

**Comparison.** Our scalp-wide alpha decrease (d = -0.20 to -0.25) is the expected arousal effect. It cannot by itself explain our claim that EEG carries valence. Without EOG or eye tracking, the frontal delta increase (+0.21) remains possibly ocular.

## 5. Validity of ultra-short HRV

- **Munoz et al. [26]** (3,387 adults; reference 240 to 300 s): a single 10-s RMSSD gave r = 0.85 to 0.86, and three averaged 10-s segments gave r = 0.94. SDNN reached about r = 0.76; they judge one 10-s RMSSD valid and recommend 30 s for SDNN.
- **Nussinovitch et al. [27]** (n = 70): mean RR and RMSSD agreed well at 10 s and at 1 min. SDNN did not.
- **Baek et al. [28]** (n = 467): minimum lengths are 10 s for heart rate, 20 s for HF power, 30 s for RMSSD, 60 s for pNN50 and 240 s for SDSD.
- **Esco and Flatt [29]** (23 athletes): agreement fell from ICC 0.98 at 60 s to 0.81 at 10 s, and they retained 60 s for lnRMSSD.

Shaffer and Ginsberg [30] list proposals of 10, 30 and 60 s, and warn that ultra-short values are not interchangeable with 5-min norms. Pecchia et al. [31] found one rigorous method among 29 papers; Shaffer et al. [32] found 3 of 28 studies with a priori limits of agreement and warn that one false beat distorts the metrics. All used resting recordings. **Comparison.** Our r = 0.94 concerns heart rate, which is valid at 10 s [28]. It does not validate 10-s RMSSD or SDNN. On this evidence, 10-s RMSSD is acceptable for group contrasts, 10-s SDNN is not, and neither is validated during emotional stimulation.

## 6. Time course and early detection

Reported latencies run from alpha desynchronization within the first second [19], to cardiac deceleration in the first 2 s [7] and reduced sway at 1 to 2 s [12], to SCR onset near 1.3 s within a 1 to 4 s window [16]. Measured second by second during a film, physiology was only modestly associated with experience [33]. Schmidt et al. [3], citing Kreibig, note that features are usually aggregated over 30 to 60 s, and that the right window length is hard to define. On DEAP, Candra et al. [34] reached 65% for valence and arousal with 3 to 12 s EEG segments, improving on longer segments. We found no early-detection study on short stimuli with subject- and stimulus-independent validation. **Comparison.** The literature puts most cardiac and electrodermal information in the first 1 to 5 s of our 10-s trials.

## 7. Validation pitfalls

Li et al. [35] showed that block designs let EEG classifiers exploit temporal autocorrelation instead of stimulus-related activity. Gerster et al. [37] report that stimulus identity, not emotion, drives EEG classification on FACED. Varoquaux [36] showed that typical sample sizes give cross-validation error bars of about ±10% for 100 samples, which the standard error across folds underestimates. **Comparison.** Our held-out people and videos and our sealed test address [4], [35] and [37]. The remaining risk is [36], so confidence intervals should come from resampling people and videos, not trials.

## 8. Habituation and stability

- **Within a session.** Heart rate, SCR and corrugator responses habituate, while startle keeps differentiating valence [38]; over up to 60 repetitions, SCR and heart rate habituate rapidly while LPP modulation remains [39].
- **Across days.** One week later, SCR modulation for emotional pictures persists but prolonged cardiac deceleration disappears for repeated pictures [7]; one day later, alpha desynchronization stays fairly stable [40].
- **Stability.** Heart rate reactivity is the most reproducible cardiovascular measure (r = 0.555), more so at shorter intervals and with three or more measurements [41]. Startle modulation over 4 weeks is moderately stable with new pictures and poorly stable with repeated ones [42].

**Comparison.** Untested so far; the literature predicts the cardiac effect fades fastest and per-person stability is moderate at best.

## What the literature suggests we test next

1. **Onset latency per signal.** Compute the within-person stress-minus-other difference per second from -5 to +10 s for each signal, bootstrapping over participants and videos. Expect alpha first, deceleration within 0 to 3 s, then SCR from about 1.3 s [7, 12, 16, 19]; use the curves to set early-detection cut-offs.
2. **Heart rate by negative emotion.** Split stress trials by video category. Deceleration should be largest for mutilation or body-boundary disgust, and weaker or reversed for fear, anger and activated sadness, while SCR stays similar across categories [9, 10, 12, 13]. A null result would agree with [9].
3. **Habituation and day effects.** Add trial order and day (1 vs 2) as moderators. Deceleration should fade faster than SCR, alpha should stay stable [7, 38, 39, 40]. Report the between-day correlation of each person's effect, expecting moderate values at best [41, 42]. First check whether videos repeat across days [42].
4. **Ultra-short HRV agreement.** Within the 10,263 sixty-second ECG windows, compare mean RR, RMSSD and SDNN from 5, 10, 20 and 30 s sub-windows with the full 60 s. Report correlation, Bland-Altman bias and limits of agreement against a tolerance fixed in advance [26, 28, 32], separately for pre-video and video segments. This tests internal agreement only, since 60 s is itself a proxy [32].
5. **Ocular control for frontal EEG.** Check whether the delta effect is largest at frontopolar sites and whether it survives adjustment for a frontopolar low-frequency blink proxy [24, 25]. Repeat the alpha contrast for stress versus excitement at matched arousal [18, 19].

## Where our study is new

- **Scale on the same trials.** 314 participants and about 23,000 trials with EEG, ECG, skin conductance and respiration recorded together. The cited studies of emotional stimulation used 15 to 50 participants [2, 10, 12, 19]; very large samples exist only for resting HRV [26].
- **Generalisation test.** We test on people and videos that are both unseen in training, with a sealed test. LOSO alone is standard [3], and the stimulus confound is documented [35, 37]. The combination seems less common (not searched systematically).
- **Matched-arousal contrast.** The cardiac deceleration itself is not new [6, 7, 10, 12]. We add its size (d about -0.1 to -0.2) under self-report labels and its use in separating stress from excitement at equal arousal.
- **What is not new.**
  - the label rule [5];
  - peripheral signals tracking arousal [9, 15];
  - alpha suppression with arousal [18, 19];
  - the validation advice [4, 36].

  Our "stress" is a self-reported negative, high-arousal reaction to a 10-s video, not a Trier Social Stress Test response [2].
- **HRV during stimulation.** The planned ultra-short HRV analysis during stimulation, not rest, would extend [26, 28].

## References

1. Giannakakis G, Grigoriadis D, Giannakaki K, Simantiraki O, Roniotis A, Tsiknakis M. Review on psychological stress detection using biosignals. IEEE Transactions on Affective Computing. 2022;13:440-460 (online 2019). doi:10.1109/TAFFC.2019.2927337
2. Schmidt P, Reiss A, Duerichen R, Marberger C, Van Laerhoven K. Introducing WESAD, a multimodal dataset for wearable stress and affect detection. Proceedings of the 20th ACM International Conference on Multimodal Interaction. 2018:400-408. doi:10.1145/3242969.3242985
3. Schmidt P, Reiss A, Dürichen R, Van Laerhoven K. Wearable-based affect recognition: a review. Sensors. 2019;19(19):4079. doi:10.3390/s19194079. PMID 31547220
4. Saeb S, Lonini L, Jayaraman A, Mohr DC, Kording KP. The need to approximate the use-case in clinical machine learning. GigaScience. 2017;6(5):1-9. doi:10.1093/gigascience/gix019. PMID 28327985
5. Giannakakis G, Grigoriadis D, Tsiknakis M. Detection of stress/anxiety state from EEG features during video watching. Annu Int Conf IEEE Eng Med Biol Soc (EMBC). 2015:6034-6037. doi:10.1109/EMBC.2015.7319767. PMID 26737667
6. Bradley MM, Codispoti M, Cuthbert BN, Lang PJ. Emotion and motivation I: defensive and appetitive reactions in picture processing. Emotion. 2001;1(3):276-298. PMID 12934687
7. Bradley MM. Natural selective attention: orienting and emotion. Psychophysiology. 2009;46(1):1-11. doi:10.1111/j.1469-8986.2008.00702.x. PMID 18778317
8. Kreibig SD. Autonomic nervous system activity in emotion: a review. Biological Psychology. 2010;84(3):394-421. doi:10.1016/j.biopsycho.2010.03.010. PMID 20371374
9. Siegel EH, Sands MK, Van den Noortgate W, Condon P, Chang Y, Dy J, Quigley KS, Barrett LF. Emotion fingerprints or emotion populations? A meta-analytic investigation of autonomic features of emotion categories. Psychological Bulletin. 2018;144(4):343-393. doi:10.1037/bul0000128. PMID 29389177
10. Palomba D, Sarlo M, Angrilli A, Mini A, Stegagno L. Cardiac responses associated with affective processing of unpleasant film stimuli. International Journal of Psychophysiology. 2000;36(1):45-57. doi:10.1016/s0167-8760(99)00099-9. PMID 10700622
11. Kreibig SD, Wilhelm FH, Roth WT, Gross JJ. Cardiovascular, electrodermal, and respiratory response patterns to fear- and sadness-inducing films. Psychophysiology. 2007;44(5):787-806. doi:10.1111/j.1469-8986.2007.00550.x. PMID 17598878
12. Hagenaars MA, Roelofs K, Stins JF. Human freezing in response to affective films. Anxiety, Stress, & Coping. 2014;27(1):27-37. doi:10.1080/10615806.2013.809420. PMID 23805855
13. Roelofs K. Freeze for action: neurobiological mechanisms in animal and human freezing. Philosophical Transactions of the Royal Society B. 2017;372(1718):20160206. doi:10.1098/rstb.2016.0206. PMID 28242739
14. Hagenaars MA, Oitzl M, Roelofs K. Updating freeze: aligning animal and human research. Neuroscience & Biobehavioral Reviews. 2014;47:165-176. doi:10.1016/j.neubiorev.2014.07.021. PMID 25108035
15. Lang PJ, Greenwald MK, Bradley MM, Hamm AO. Looking at pictures: affective, facial, visceral, and behavioral reactions. Psychophysiology. 1993;30(3):261-273. doi:10.1111/j.1469-8986.1993.tb03352.x. PMID 8497555
16. Benedek M, Kaernbach C. A continuous measure of phasic electrodermal activity. Journal of Neuroscience Methods. 2010;190(1):80-91. doi:10.1016/j.jneumeth.2010.04.028. PMID 20451556
17. Sjouwerman R, Lonsdorf TB. Latency of skin conductance responses across stimulus modalities. Psychophysiology. 2019;56(4):e13307. doi:10.1111/psyp.13307. PMID 30461024
18. De Cesarei A, Codispoti M. Affective modulation of the LPP and alpha-ERD during picture viewing. Psychophysiology. 2011;48(10):1397-1404. doi:10.1111/j.1469-8986.2011.01204.x. PMID 21486291
19. Schubring D, Schupp HT. Affective picture processing: alpha- and lower beta-band desynchronization reflects emotional arousal. Psychophysiology. 2019;56(8):e13386. doi:10.1111/psyp.13386. PMID 31026079
20. Reznik SJ, Allen JJB. Frontal asymmetry as a mediator and moderator of emotion: an updated review. Psychophysiology. 2018;55(1):e12965. doi:10.1111/psyp.12965. PMID 28776710
21. van der Vinne N, Vollebregt MA, van Putten MJAM, Arns M. Frontal alpha asymmetry as a diagnostic marker in depression: fact or fiction? A meta-analysis. NeuroImage: Clinical. 2017;16:79-87. doi:10.1016/j.nicl.2017.07.006. PMID 28761811
22. Katmah R, Al-Shargie F, Tariq U, Babiloni F, Al-Mughairbi F, Al-Nashash H. A review on mental stress assessment methods using EEG signals. Sensors. 2021;21(15):5043. doi:10.3390/s21155043. PMID 34372280
23. Knyazev GG. EEG delta oscillations as a correlate of basic homeostatic and motivational processes. Neuroscience & Biobehavioral Reviews. 2012;36(1):677-695. doi:10.1016/j.neubiorev.2011.10.002. PMID 22020231
24. Hagemann D, Naumann E. The effects of ocular artifacts on (lateralized) broadband power in the EEG. Clinical Neurophysiology. 2001;112(2):215-231. doi:10.1016/s1388-2457(00)00541-1. PMID 11165523
25. Plöchl M, Ossandón JP, König P. Combining EEG and eye tracking: identification, characterization, and correction of eye movement artifacts in electroencephalographic data. Frontiers in Human Neuroscience. 2012;6:278. doi:10.3389/fnhum.2012.00278. PMID 23087632
26. Munoz ML, van Roon A, Riese H, Thio C, Oostenbroek E, Westrik I, de Geus EJ, Gansevoort R, Lefrandt J, Nolte IM, Snieder H. Validity of (ultra-)short recordings for heart rate variability measurements. PLoS ONE. 2015;10(9):e0138921. doi:10.1371/journal.pone.0138921. PMID 26414314
27. Nussinovitch U, Elishkevitz KP, Katz K, Nussinovitch M, Segev S, Volovitz B, Nussinovitch N. Reliability of ultra-short ECG indices for heart rate variability. Annals of Noninvasive Electrocardiology. 2011;16(2):117-122. doi:10.1111/j.1542-474X.2011.00417.x. PMID 21496161
28. Baek HJ, Cho CH, Cho J, Woo JM. Reliability of ultra-short-term analysis as a surrogate of standard 5-min analysis of heart rate variability. Telemedicine and e-Health. 2015;21(5):404-414. doi:10.1089/tmj.2014.0104. PMID 25807067
29. Esco MR, Flatt AA. Ultra-short-term heart rate variability indexes at rest and post-exercise in athletes: evaluating the agreement with accepted recommendations. Journal of Sports Science and Medicine. 2014;13(3):535-541. PMID 25177179
30. Shaffer F, Ginsberg JP. An overview of heart rate variability metrics and norms. Frontiers in Public Health. 2017;5:258. doi:10.3389/fpubh.2017.00258. PMID 29034226
31. Pecchia L, Castaldo R, Montesinos L, Melillo P. Are ultra-short heart rate variability features good surrogates of short-term ones? State-of-the-art review and recommendations. Healthcare Technology Letters. 2018;5(3):94-100. doi:10.1049/htl.2017.0090. PMID 29922478
32. Shaffer F, Meehan ZM, Zerr CL. A critical review of ultra-short-term heart rate variability norms research. Frontiers in Neuroscience. 2020;14:594880. doi:10.3389/fnins.2020.594880. PMID 33328866
33. Mauss IB, Levenson RW, McCarter L, Wilhelm FH, Gross JJ. The tie that binds? Coherence among emotion experience, behavior, and physiology. Emotion. 2005;5(2):175-190. doi:10.1037/1528-3542.5.2.175. PMID 15982083
34. Candra H, Yuwono M, Chai R, Handojoseno A, Elamvazuthi I, Nguyen HT, Su S. Investigation of window size in classification of EEG-emotion signal with wavelet entropy and support vector machine. Annu Int Conf IEEE Eng Med Biol Soc (EMBC). 2015:7250-7253. doi:10.1109/EMBC.2015.7320065. PMID 26737965
35. Li R, Johansen JS, Ahmed H, Ilyevsky TV, Wilbur RB, Bharadwaj HM, Siskind JM. The perils and pitfalls of block design for EEG classification experiments. IEEE Transactions on Pattern Analysis and Machine Intelligence. 2021;43:316-333. doi:10.1109/TPAMI.2020.2973153. PMID 33211652
36. Varoquaux G. Cross-validation failure: small sample sizes lead to large error bars. NeuroImage. 2018;180(Pt A):68-77. doi:10.1016/j.neuroimage.2017.06.061. PMID 28655633
37. Gerster M, Sirotina E, Orlovskii A, Hertz A, Champaud J, et al. Stimulus identity rather than emotion drives EEG classification on the FACED dataset. bioRxiv preprint. 2026. doi:10.64898/2026.06.12.731889
38. Bradley MM, Lang PJ, Cuthbert BN. Emotion, novelty, and the startle reflex: habituation in humans. Behavioral Neuroscience. 1993;107(6):970-980. doi:10.1037//0735-7044.107.6.970. PMID 8136072
39. Codispoti M, Ferrari V, Bradley MM. Repetitive picture processing: autonomic and cortical correlates. Brain Research. 2006;1068(1):213-220. doi:10.1016/j.brainres.2005.11.009. PMID 16403475
40. Ferrari V, Mastria S, Codispoti M. The interplay between attention and long-term memory in affective habituation. Psychophysiology. 2020;57(6):e13572. doi:10.1111/psyp.13572. PMID 32239721
41. Swain A, Suls J. Reproducibility of blood pressure and heart rate reactivity: a meta-analysis. Psychophysiology. 1996;33(2):162-174. doi:10.1111/j.1469-8986.1996.tb02120.x. PMID 8851244
42. Larson CL, Ruffalo D, Nietert JY, Davidson RJ. Temporal stability of the emotion-modulated startle response. Psychophysiology. 2000;37(1):92-101. PMID 10705771
