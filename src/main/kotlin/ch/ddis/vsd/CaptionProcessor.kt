package ch.ddis.vsd

import edu.stanford.nlp.ling.CoreLabel
import edu.stanford.nlp.pipeline.CoreDocument
import edu.stanford.nlp.pipeline.StanfordCoreNLP
import java.util.*

object CaptionProcessor {

    private val pipeline: StanfordCoreNLP

    val relevantWordTags = setOf(
        "JJ",   //Adjective
        "JJR",  //Adjective, comparative
        "JJS",  //Adjective, superlative
        "NN",   //Noun, singular or mass
        "NNS",  //Noun, plural
        "NNP",  //Proper noun, singular
        "NNPS", //Proper noun, plural
        "RB",   //Adverb
        "RBR",  //Adverb, comparative
        "RBS",  //Adverb, comparative
        "VB",   //Verb, base form
        "VBD",  //Verb, past tense
        "VBG",  //Verb, gerund or present participle
        "VBN",  //Verb, past participle
        "VBP",  //Verb, non3rd person singular present
        "VBZ",  //Verb, 3rd person singular present

    )

    init {
        val props = Properties()
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,depparse")
        pipeline = StanfordCoreNLP(props)
    }

    fun getRelevantTokens(string: String): Set<CoreLabel> {
        val document = CoreDocument(string)
        pipeline.annotate(document)
        return document.tokens().filter { it.tag() in relevantWordTags && (it.lemma() != "be") }.toSet()
    }



}