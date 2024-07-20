package ch.ddis.vsd

import com.fasterxml.jackson.core.type.TypeReference
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import java.io.File
import java.io.FileWriter
import java.io.PrintWriter


object Main {

    @JvmStatic
    fun main(args: Array<String>) {

        val mapper = jacksonObjectMapper()

        val paragraphs = mapper.readValue(
            File("data/stanford_image_paragraphs/paragraphs_v1.json"),
            object : TypeReference<List<StanfordParagraphEntry>>() {})

        File("data/count/stanford_paragraphs_counts.csv").printWriter().use { writer ->

            writer.println("id,count")

            paragraphs.forEach {
                val count = CaptionProcessor.getRelevantTokens(it.paragraph).size
                writer.println("${it.image_id},$count")
            }

        }


        val files = File("data/localized_narratives").listFiles { _, name -> name.endsWith(".jsonl") } ?: emptyArray()

        files.forEach { inputFile ->
            val entries = inputFile.readLines().map { mapper.readValue(it, LocalizedNarrativesEntry::class.java) }
            val outFile = File("data/count/${inputFile.nameWithoutExtension}_counts.csv")

            var skip = 0

            if (outFile.exists()) {
                skip = outFile.readLines().size - 1
                if (skip >= entries.size) {
                    return@forEach
                }
            }

            val writer = PrintWriter(FileWriter(outFile, true))

            if (skip == 0) {
                writer.println("id,count")
            }

            println("starting ${inputFile.name}")

            var counter = skip
            entries.drop(skip).forEach {
                val count = CaptionProcessor.getRelevantTokens(it.caption).size
                writer.println("${it.image_id},$count")
                if (++counter % 100 == 0) {
                    writer.flush()
                    println("written $counter entries")
                }
            }
            writer.flush()
            println("written $counter entries")
            writer.close()
            println("done with ${inputFile.name}")
            println()
        }

    }

}