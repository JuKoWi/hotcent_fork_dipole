{
	inputs = { 
		nixpkgs.url = "nixpkgs/nixos-unstable";
	};

	outputs = { self, nixpkgs }:
		let 
        		pkgs = import nixpkgs {
                		system="x86_64-linux";
                 		config.allowUnfree = true;
        		};
	     		qc-grid = p: p.buildPythonPackage rec {
				pname = "qc-grid";
				version = "0.0.9";
				pyproject = true;
				src = pkgs.fetchFromGitHub { 
					owner = "theochem";
					repo = "grid";
					rev = "v${version}";
					hash = "sha256-xbgA84H6QffqnTRw/x0/oKS46vNc00VyN60NGv7nxsk=";
				};
				build-system = with p; [ setuptools setuptools-scm ];
				dependencies = with p; [ numpy scipy sympy pytest importlib-resources ];
				doCheck = false;
				pythonImportsCheck = ["grid"];
			};
             		myPython = pkgs.python3.withPackages (p: with p; [
		    		ase
		    		matplotlib
		    		numpy
		    		pytest
		    		pyyaml
		    		scipy
		    		sympy
		    		libxc
		    		ipython
		    		jupyter
		    		ipykernel
		    		snakeviz
				(qc-grid p)
             		]);
        in {
        	devShell.x86_64-linux = pkgs.mkShell {
                	buildInputs = [
                        	myPython
			   	pkgs.ruff
                    	];
			shellHook = ''ROOT_PATH=$(git rev-parse --show-toplevel) export PYTHONPATH="$ROOT_PATH:$PYTHONPATH";  # or "PYTHONPATH=./" if using mkShell rec'';
                };

	};
}
