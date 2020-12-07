# Docker Deployment Action

A [GitHub Action](https://github.com/marketplace/actions/docker-deployment) that supports docker-compose and Docker Swarm deployments.

## Examples

Below is a brief examples on how the action can be used:

```yaml tab="Swarm"
- name: Deploy to Docker swarm
  uses: wshihadeh/docker-deployment-action@v1
  with:
    remote_docker_host: user@myswarm.com
    ssh_private_key: ${{ secrets.DOCKER_SSH_PRIVATE_KEY }}
    ssh_public_key: ${{ secrets.DOCKER_SSH_PUBLIC_KEY }}
    deployment_mode: docker-swarm
    args: my_applicaion
```

```yaml tab="Compose"
- name: Deploy to Docker Host
  uses: wshihadeh/docker-deployment-action@v1
  with:
    remote_docker_host: user@myswarm.com
    ssh_private_key: ${{ secrets.DOCKER_SSH_PRIVATE_KEY }}
    ssh_public_key: ${{ secrets.DOCKER_SSH_PUBLIC_KEY }}
    deployment_mode: docker-compose
    args: up -d
    pre_deployment_command_args: 'bundle exec rake db:migrate'
    docker_prune: 'true'
    pull_images_first: 'true'
```

```yaml tab="Compose with copy"
- name: Deploy to Docker Host
  uses: wshihadeh/docker-deployment-action@v1
  with:
    remote_docker_host: user@myswarm.com
    ssh_private_key: ${{ secrets.DOCKER_SSH_PRIVATE_KEY }}
    ssh_public_key: ${{ secrets.DOCKER_SSH_PUBLIC_KEY }}
    deployment_mode: docker-compose
    copy_stack_file: true
    deploy_path: /root/my-deployment
    stack_file_name: docker-compose.yaml
    keep_files: 5
    args: up -d
    docker_prune: 'false'
    pull_images_first: 'false'
```

```yaml tab="Swarm with copy"
- name: Deploy to Docker swarm
  uses: wshihadeh/docker-deployment-action@v1
  with:
    remote_docker_host: user@myswarm.com
    ssh_private_key: ${{ secrets.DOCKER_SSH_PRIVATE_KEY }}
    ssh_public_key: ${{ secrets.DOCKER_SSH_PUBLIC_KEY }}
    deployment_mode: docker-swarm
    copy_stack_file: true
    deploy_path: /root/my-deployment
    stack_file_name: docker-stack.yaml
    keep_files: 5
    args: my_applicaion
```

## Input Configurations

Below are all of the supported inputs. Some inputs are considered sensitive information and it should be stored as secrets.

### `args`

Arguments to pass to the deployment command either  `docker`  or `docker-compose`. The actions will automatically generate the follwing commands for each of the cases.

- `docker stack deploy --compose-file $FILE --log-level debug --host $HOST`
- `docker-compose -f $INPUT_STACK_FILE_NAME`

### `remote_docker_host`

Specify Remote Docker host. The input value must be in the follwing format (user@host)

### `ssh_public_key`

Remote Docker SSH public key.

### `ssh_private_key`

SSH private key used to connect to the docker host

### `deployment_mode`

Deployment mode either docker-swarm or docker-compose. Default is docker-compose.

### `copy_stack_file`

Copy stack file to remote server and deploy from the server. Default is false.

### `deploy_path`

The path where the stack files will be copied to. Default ~/docker-deployment.

### `stack_file_name`

Docker stack file used. Default is docker-compose.yaml

### `keep_files`

Number of the files to be kept on the server. Default is 3.

### `docker_prune`

A boolean input to trigger docker prune command.

### `pre_deployment_command_args`

The args for the pre deploument command. Applicable only for docker-compose.

### `pull_images_first`

Pull docker images before deploying. Applicable only for docker-compose.

## License

This project is licensed under the MIT license. See the [LICENSE](LICENSE) file for details.
