from mario_gpt import SampleOutput

generated_level = SampleOutput.load("generated_level.txt")

# play in interactive
generated_level.play()

# run Astar agent
r = generated_level.run_astar_evaluate()

print(r.stdout)