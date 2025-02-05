#!/usr/bin/env python3
import os
import sys
import json
import yaml
import re
from pathlib import Path
from typing import Dict, List, Set

class DockerValidator:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.errors = []
        self.warnings = []

    def validate_all(self) -> bool:
        """Run all validations and return True if no errors were found."""
        self.validate_docker_compose()
        self.validate_dockerfiles()
        self.validate_dependencies()
        self.validate_environment_variables()
        self.validate_ports()
        
        if self.errors:
            print("\n❌ Validation failed with the following errors:")
            for error in self.errors:
                print(f"  • {error}")
        if self.warnings:
            print("\n⚠️ Warnings:")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        return len(self.errors) == 0

    def validate_docker_compose(self):
        """Validate docker-compose.yml configuration."""
        compose_path = self.project_root / "docker-compose.yml"
        if not compose_path.exists():
            self.errors.append("docker-compose.yml not found")
            return

        try:
            with open(compose_path) as f:
                compose_config = yaml.safe_load(f)

            services = compose_config.get('services', {})
            defined_volumes = set(compose_config.get('volumes', {}).keys())
            used_volumes = set()
            defined_ports = set()

            for service_name, service in services.items():
                # Check for required fields
                if 'build' not in service and 'image' not in service:
                    self.errors.append(f"Service {service_name} missing both 'build' and 'image' directives")

                # Validate volumes
                for volume in service.get('volumes', []):
                    if ':' in volume:
                        host_path = volume.split(':')[0]
                        if host_path not in defined_volumes and not Path(host_path).exists():
                            self.warnings.append(f"Volume host path '{host_path}' in service '{service_name}' does not exist")
                    else:
                        used_volumes.add(volume)

                # Check for port conflicts
                for port_mapping in service.get('ports', []):
                    host_port = port_mapping.split(':')[0]
                    if host_port in defined_ports:
                        self.errors.append(f"Port {host_port} is mapped multiple times")
                    defined_ports.add(host_port)

                # Validate healthcheck
                healthcheck = service.get('healthcheck', {})
                if healthcheck:
                    if 'test' not in healthcheck:
                        self.errors.append(f"Service {service_name} has healthcheck but no test command")
                    if 'interval' not in healthcheck:
                        self.warnings.append(f"Service {service_name} healthcheck missing interval")

        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML in docker-compose.yml: {str(e)}")
        except Exception as e:
            self.errors.append(f"Error validating docker-compose.yml: {str(e)}")

    def validate_dockerfiles(self):
        """Validate Dockerfile configurations."""
        dockerfile_paths = list(self.project_root.glob("**/Dockerfile"))
        
        for dockerfile in dockerfile_paths:
            try:
                with open(dockerfile) as f:
                    content = f.read()

                # Check for best practices
                if 'FROM' not in content:
                    self.errors.append(f"{dockerfile} missing FROM instruction")

                if content.count('FROM') > 1 and 'AS' not in content:
                    self.warnings.append(f"{dockerfile} has multiple FROM instructions but isn't using multi-stage builds")

                if 'ADD' in content and not re.search(r'ADD\s+https?://', content):
                    self.warnings.append(f"{dockerfile} uses ADD instead of COPY for local files")

                if not re.search(r'WORKDIR\s+/', content):
                    self.warnings.append(f"{dockerfile} missing WORKDIR instruction")

                # Check for common security issues
                if 'sudo' in content:
                    self.warnings.append(f"{dockerfile} contains sudo command")

                if re.search(r'apt-get\s+install(?!.*--no-install-recommends)', content):
                    self.warnings.append(f"{dockerfile} installs packages without --no-install-recommends")

            except Exception as e:
                self.errors.append(f"Error validating {dockerfile}: {str(e)}")

    def validate_dependencies(self):
        """Validate Python and Node.js dependencies."""
        # Validate Python requirements
        for req_file in self.project_root.glob("**/requirements.txt"):
            try:
                with open(req_file) as f:
                    requirements = f.read()
                
                # Check for version pinning
                for line in requirements.splitlines():
                    if line and not line.startswith('#'):
                        if not re.search(r'[=>~=]=|\d+\.\d+\.\d+', line):
                            self.warnings.append(f"Dependency in {req_file} not version-pinned: {line}")

            except Exception as e:
                self.errors.append(f"Error validating {req_file}: {str(e)}")

        # Validate package.json
        for package_json in self.project_root.glob("**/package.json"):
            try:
                with open(package_json) as f:
                    pkg_data = json.load(f)
                
                deps = {**pkg_data.get('dependencies', {}), **pkg_data.get('devDependencies', {})}
                for pkg, version in deps.items():
                    if version.startswith('^') or version.startswith('~'):
                        self.warnings.append(f"Package {pkg} in {package_json} uses loose version: {version}")

            except json.JSONDecodeError as e:
                self.errors.append(f"Invalid JSON in {package_json}: {str(e)}")
            except Exception as e:
                self.errors.append(f"Error validating {package_json}: {str(e)}")

    def validate_environment_variables(self):
        """Validate environment variables configuration."""
        compose_path = self.project_root / "docker-compose.yml"
        if not compose_path.exists():
            return

        try:
            with open(compose_path) as f:
                compose_config = yaml.safe_load(f)

            for service_name, service in compose_config.get('services', {}).items():
                # Check environment variables
                env_vars = service.get('environment', {})
                if isinstance(env_vars, list):
                    for env in env_vars:
                        if '=' not in env:
                            self.warnings.append(f"Service {service_name} has environment variable without value: {env}")
                elif isinstance(env_vars, dict):
                    for key, value in env_vars.items():
                        if value is None:
                            self.warnings.append(f"Service {service_name} has environment variable without value: {key}")

        except Exception as e:
            self.errors.append(f"Error validating environment variables: {str(e)}")

    def validate_ports(self):
        """Validate port configurations."""
        used_ports = set()
        compose_path = self.project_root / "docker-compose.yml"
        
        if not compose_path.exists():
            return

        try:
            with open(compose_path) as f:
                compose_config = yaml.safe_load(f)

            for service_name, service in compose_config.get('services', {}).items():
                for port_mapping in service.get('ports', []):
                    if isinstance(port_mapping, str):
                        host_port = port_mapping.split(':')[0]
                        if host_port in used_ports:
                            self.errors.append(f"Port conflict detected: {host_port} is used multiple times")
                        used_ports.add(host_port)

        except Exception as e:
            self.errors.append(f"Error validating ports: {str(e)}")

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    validator = DockerValidator(project_root)
    
    print("🔍 Validating Docker configuration...")
    is_valid = validator.validate_all()
    
    if is_valid:
        print("\n✅ Validation passed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Validation failed. Please fix the errors above before building.")
        sys.exit(1)

if __name__ == "__main__":
    main()
