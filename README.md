# Train-Routes-Graph-Search 

Testing different graph search algorithms on major U.S. train routes

**File Descriptions** 

rrEdges.txt : Each line connects the two given node IDs

rrNodeCity.txt : Each line has an ID corresponding to a major city

rrNodes.txt : Each line has an ID corresponding to a latitude and longitude

Searches.py : File that tests the performances of different search algorithms (BFS, Bidirectional BFS, A*, Bidirectional A*, and Tridirectional A*) and visualizes them in Tkinter 

Data is originally from: http://cta.ornl.gov/transnet/ (outdated link)

**A few notes**
- You may only use the major cities in the rrNodesCity.txt file (working on adding more cities)
- You have to input the start, target, and intermediate city (intermediate is for tridirectional search)

<img width="877" alt="bfsusa" src="https://github.com/user-attachments/assets/730ae48f-f8c6-41e4-85a8-d1671f61737c">
