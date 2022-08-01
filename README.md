INSTALLING THE EGRAPHSEN ANNOTATION TOOL

2. Use command line to create a conda environment via (substitute "name", e.g. labelme but make sure the environment doesnt exist already. You can check existing environments via "conda env list"): 
	"conda create -n name python=3"
3. Activate the environment with 
	"conda activate name" (where name is the name given to the environment)
4. Navigate to the "egraphsen-tool" directory via "cd" and "dir"
5. In the folder "egraphsen-tool", run the command
	"pip install -r requirements.txt"
6. Start the Tool by running "python run.py"	