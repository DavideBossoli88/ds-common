#!/usr/bin/env bash

docker build -t train_automl --build-arg username_proxy=null --build-arg password_proxy=null --build-arg git_token=<token> .