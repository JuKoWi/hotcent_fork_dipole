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

             myPython = pkgs.python3.withPackages (p: with p; [
		    ase
		    matplotlib
		    numpy
		    pytest
		    pyyaml
		    scipy
		    sympy
		    libxc
             ]);

#	     libxc = pkgs.libxc;

        in {
           devShell.x86_64-linux =
                pkgs.mkShell {
                    buildInputs = [
                           myPython
#			   libxc
                           # pkgs.gtest
                    ];
shellHook = ''
                    ROOT_PATH=$(git rev-parse --show-toplevel)
		    export PYTHONPATH="$ROOT_PATH:$PYTHONPATH";  # or "PYTHONPATH=./" if using mkShell rec
            '';
                };

    };
}
