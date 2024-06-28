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
// const parseCodeBlockUsingRegex = (text) => {
//   const codeBlockRegex = /```(.*?)```/s;
//   const match = text.match(codeBlockRegex);
//   return match ? match[1].trim() : "No code block found";
// };

// Prompt Template
const prompt = ChatPromptTemplate.fromMessages([
  `
    Answer the User Question.
    Context: {context}
    Question: {input}    
`,
]);

// Document Loader
const loader = new CheerioWebBaseLoader("https://www.rust-lang.org/");

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
    input: "What is Rust?",
  });

  const response = JSON.stringify(retrievedResponse, null, 2);

  //   const message = response.text;

  //   const parsedResponse = message;
  console.log(`\n${response}\n`);
};

run();
