from agent_v5 import AgentBase, AnnotationAgent, ResultAgent, PlanningAgent
import time

start_time = time.time()
planning_agent = PlanningAgent()
planning_agent.build_graph()
planning_agent.graph.execute_graph()
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Execution time: {elapsed_time:.2f} seconds")

