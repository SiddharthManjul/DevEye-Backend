import * as dotenv from "dotenv";
dotenv.config();

import { TaskType } from "@google/generative-ai";

import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrievalChain } from "langchain/chains/retrieval";

import promptSync from 'prompt-sync';
const promptFromTerminal = promptSync({sigint: true});

// Model Initialization
const model = new ChatGoogleGenerativeAI({
  apiKey: process.env.GEMINI_API_KEY,
  model: "gemini-pro",
  temperature: 0,
  // maxOutputTokens: 2048,
});

// String Parser from Langchain
const parser = new StringOutputParser();

// Regex Parser
// const codeBlockRegex = /```(.*?)```/s;
// const regexCodeBlockParser = new RegexParser(codeBlockRegex, 1);

// Custom Regex Parser
const parseCodeBlockUsingRegex = (text) => {
  const codeBlockRegex = /```(.*?)```/s;
  const match = text.match(codeBlockRegex);
  if (match) {
    // Replace \n with actual newline characters
    return match[1].replace(/\\n/g, '\n').replace(/\\"/g, '"').trim();
  }
  return "No code block found";
};

// Prompt Template
const prompt = ChatPromptTemplate.fromMessages([
  `
    Answer the User Question.
    Context: {context}
    Question: {input}    
`,
]);

// Document Loader
const loader = new CheerioWebBaseLoader("https://doc.rust-lang.org/book/ch01-01-installation.html");

const docs = await loader.load();

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 2000,
  chunkOverlap: 200,
});

const splitDocs = await splitter.splitDocuments(docs);

const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GEMINI_API_KEY,
  model: "embedding-001",
  taskType: TaskType.RETRIEVEL_DOCUMENT,
});

const vecotrStore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings
);

// Retrieve Data
const retriever = vecotrStore.asRetriever({
    k: 0
});

const input = promptFromTerminal("How can I help you? ");

// Runner Function
const run = async () => {
  // const chain = prompt.pipe(model);
  const chain = await createStuffDocumentsChain({
    llm: model,
    prompt: prompt,
  });

  const parsedChain = chain.pipe(parser);

  const retrievalChain = await createRetrievalChain({
    combineDocsChain: parsedChain,
    retriever: retriever,
  });

  const retrievedResponse = await retrievalChain.invoke({
    input: input,
  });

  const response = JSON.stringify(retrievedResponse, null, 2);
  console.log(response);

  // const message = retrievedResponse;

  const parsedResponse = parseCodeBlockUsingRegex(response);
  console.log(`\n${parsedResponse}\n`);
};

run();
