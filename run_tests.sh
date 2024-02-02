#!/bin/bash 

# test script shared for local dev and CI

coverage run -m pytest
coverage report