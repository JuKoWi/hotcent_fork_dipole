{
  description = "hotcent";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python3;

        hotcent = python.pkgs.buildPythonPackage {
          pname = "hotcent";
          version = "2.0.1";
          src = ./.;
          format = "setuptools";

          propagatedBuildInputs = with python.pkgs; [
            ase
            matplotlib
            numpy
            pytest
            pyyaml
            scipy
          ];

          pythonImportsCheck = [ "hotcent" ];

          meta = with pkgs.lib; {
            description = "hotcent";
            homepage = "https://gitlab.com/mvdb/hotcent";
            license = licenses.gpl3Plus;
            maintainers = with maintainers; [ mvdb ];
          };
        };

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [ hotcent ];
        };
        packages.default = python.withPackages (p: [ hotcent ]);
        formatter = pkgs.nixpkgs-fmt;
      }
    );
}
