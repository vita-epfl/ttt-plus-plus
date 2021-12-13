import numpy as np
import torch
from utils.misc import *
from test_calls.show_result import get_err_adapted

corruptions_names = ['snow']

corruptions = ['snow']

info = []
info.append(('gn', '_expand', 5))

########################################################################

def compare_results(table, prec1=True):
	print("%.2f \t %.2f" % (table[0], table[1]))

def show_table(folder, level, threshold):
	results = []
	for corruption in corruptions:
		row = []
		try:
			rdict_ada = torch.load(folder + '/%s_%d_ada.pth' %(corruption, level))
			rdict_inl = torch.load(folder + '/%s_%d_inl.pth' %(corruption, level))

			ssh_confide = rdict_ada['ssh_confide']
			new_correct = rdict_ada['cls_correct']
			old_correct = rdict_inl['cls_correct']

			row.append(rdict_inl['cls_initial'])
			old_correct = old_correct[:len(new_correct)]
			err_adapted = get_err_adapted(new_correct, old_correct, ssh_confide, threshold=threshold)
			row.append(err_adapted)

		except:
			row.append(0)
			row.append(0)
		results.append(row)

	results = np.asarray(results)
	results = np.transpose(results)
	results = results * 100
	return results

def show_none(folder, level):
	results = []
	for corruption in corruptions:
		try:
			rdict_inl = torch.load(folder + '/%s_%d_none.pth' %(corruption, level))
			results.append(rdict_inl['cls_initial'])
		except:
			results.append(0)
	results = np.asarray([results])
	results = results * 100
	return results

for parta, partb, level in info:
	print("\n----- Setting ----- \n")

	print("Level: #{:d} \nNet: {}".format(level, parta + partb))

	# TODO: when to terminate?
	if parta == 'bn':
		threshold = 0.9
	else:
		threshold = 1

	print("\n----- Layer #2 ----- \n")

	print("Test \t Adapt")
	results_slow = show_table('results/C10C_layer2_%s_%s%s' %('slow', parta, partb), level, threshold=threshold)
	compare_results(results_slow)

	results_onln = show_table('results/C10C_layer2_%s_%s%s' %('online', parta, partb), level, threshold=threshold)
	compare_results(results_onln)

	print("\n----- Layer #3 ----- \n")

	print("Test \t Adapt")
	results_slow = show_table('results/C10C_layer3_%s_%s%s' %('slow', parta, partb), level, threshold=threshold)
	compare_results(results_slow)

	results_onln = show_table('results/C10C_layer3_%s_%s%s' %('online', parta, partb), level, threshold=threshold)
	compare_results(results_onln)
