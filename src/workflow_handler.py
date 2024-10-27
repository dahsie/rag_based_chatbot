
from vectorstore import VectorStore
from graph_state import GraphState
from typing import Dict
from json.decoder import JSONDecodeError
from langchain_core.output_parsers.json import OutputParserException
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from pprint import pprint
from utils import extract_dict
from utils import parse_web_search_result

class WorkflowHandler:

    def __init__(self, vectorstore: VectorStore,rag_chain, grader_parser, retrieval_grader, question_rewriter, web_search_tool, state: Dict =  {}):

        self.state = state
        self.__vectorstore = vectorstore
        self.__grader_parser = grader_parser

        #Chains definition
        self.__rag_chain = rag_chain
        self.__retrieval_grader = retrieval_grader
        self.__question_rewriter = question_rewriter

        #web search tool instanciation
        self.__web_search_tool = web_search_tool

        self.__app = None

    def retrieve(self,state: Dict) -> Dict:
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = self.__vectorstore.retriever.invoke(question)
        return {"documents": documents, "question": question}
    
    def generate(self, state: Dict) -> Dict:
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = self.__rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}
    
    def grade_documents(self,state: Dict) -> Dict:
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        web_search = "No"
        for d in documents:
            score = self.__retrieval_grader.invoke({"question": question, "document": d.page_content})

            if score != '' :

                print(f" score :{score}")

                try :
                    score = self.__grader_parser.parse(score)
                except (JSONDecodeError,OutputParserException):
                    score = extract_dict(score)
                    
                grade = score.get("score", "no")
                    
                print(f"grade : {grade}")
            else :
                grade = 'no'
                
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
                continue
        return {"documents": filtered_docs, "question": question, "web_search": web_search}
    

    def transform_query(self, state: Dict) -> Dict:
        """

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        # better_question = question_rewriter.invoke({"question": question})
        response = self.__question_rewriter.invoke({"question": question})
        better_question = response['question']
        return {"documents": documents, "question": better_question}


    def web_search(self,state: Dict) -> Dict:
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        print("---WEB SEARCH---")
        question = state["question"]
        documents = state["documents"]

        # Web search
        docs = self.__web_search_tool.invoke({"query": question})

        if isinstance(docs, str):
            docs = parse_web_search_result(docs)
        print(f" question : {question}")

        web_results = "\n".join([d["snippet"] for d in docs])
        # web_results = "\n".join([d["content"] for index, d in enumerate(docs)])
        web_results = Document(page_content=web_results)
        documents.append(web_results)

        # print(f"document : {documents} -- question : {question}")
        return {"documents": documents, "question": question}



    def decide_to_generate(self, state: Dict) -> str:
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        state["question"]
        web_search = state["web_search"]
        state["documents"]

        if web_search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"
        
        
    def build_workflow(self):
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("retrieve", self.retrieve)  # retrieve
        workflow.add_node("grade_documents", self.grade_documents)  # grade documents
        workflow.add_node("generate", self.generate)  # generatae
        workflow.add_node("transform_query", self.transform_query)  # transform_query
        workflow.add_node("web_search_node", self.web_search)  # web search

        # Build graph
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "web_search_node")
        workflow.add_edge("web_search_node", "generate")
        workflow.add_edge("generate", END)

        self.__app = workflow.compile()

    def invoke(self, inputs : Dict):   

        # Run
        # inputs = {"question": "How does the AlphaCodium paper work?"}
        for output in self.__app.stream(inputs):
            # print(f" output : {output}")
            for key, value in output.items():
                # Node
                pprint(f"Node '{key}' :")
                # Optional: print full state at each node
                # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
            pprint("\n---\n")

        # Final generation
        pprint(value["generation"])