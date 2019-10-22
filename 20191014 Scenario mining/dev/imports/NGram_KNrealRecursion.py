# n-gram class
#refs:  http://www.decontextualize.com/teaching/dwwp/topics-n-grams-and-markov-chains/
#		https://lagunita.stanford.edu/c4x/Engineering/CS-224N/asset/slp4.pdf
#		chen and goodman 2008 - an empirical study of smoothing techniques for language modelling

# history (unfortunately partial because i started only on 20171101 to write it
# 20171101		added the possibility to walk through the model. added some other utilities e.g. get trainingDataSize
# 201711		implemented ful recursion of KN smoothing (which led to a huge cleanup)
# 20180118		added the possibility to rely on a fullVocabualrySize across all models
# 20180314		removed fullVocabularySize as static variable and made it object dependent
# 20180515		checked the difference on how to deal with fullVocabualrySize, could not find much
#				some cleanup done (removed commented code)
# 20180525		added the possibility to rename the start and end node

				

# TODO
# check that the calculations are in accordance with the papers!

import pandas as pd
from warnings import warn
from graphviz import Digraph
from warnings import warn
import numpy as np
from copy import deepcopy

class recursive_NGramKneserNey(object) :

	''' this class implements ngrams and smoothing with Kneser Ney. 
		any n is possible for ngrams, but at the moment this class is tailored for trigrams.. maybe..
		it implements the chain probability calculation of the trigram (quadrigram etc not implemented) via full recursion
		OOV words are therefore modelled as said in Jurafsy-martin, i.e. unigram with count zero, 
		so that P_KN(unk) = lambda(epsilon)/|V|
		lambda calculation is now omtited
		
		question: do we need to keep unigrams etc of <start> and <stop> at all!?	(we do?)
		what about self.beginnings and self.ends? (i think we need them for when we walk)
		what about lambda(epsilon)? 
	'''
	
	startToken = "<start>"
	stopToken = "<stop>"
	unknownToken = "<unk>"
	
	def __init__(self, name, n=2, d=0.75, lambdaEpsilon=0.75, modelOOVfromTrainingData=False, useFixedVocabularySize=False, fullVocabularySize=0, printTodo=True) :
		self.name = name
		self.n = n
		self.d = d
		self.lambdaEpsilon = lambdaEpsilon
		self.modelOOVfromTrainingData = modelOOVfromTrainingData
		self.useFixedVocabularySize = useFixedVocabularySize
		self.fullVocabularySize = fullVocabularySize
		self.trainingData = []
		self.observedVocabulary = []
		
		self.grams = {}  # a dict of dict, where the key here represents "n"
		
		for i in range(1, self.n+1) :
			self.grams[i] = {}
		
		if self.modelOOVfromTrainingData == False :
			self.grams[1][(self.unknownToken,)] = 0
		
		if self.n > 3 :
			if printTodo :
				print("for n>3 we MUST CHECK!")
				self.printTODO()
				
	def getTrainingDataSize(self) :
		return len(self.trainingData)
		
	def getUniqueTrainingDataSize(self) :
		return len(set(self.trainingData))
		
	def printGramInfo(self) :
		print("---------------------------")
		print("gram", self.name)
		print("n =", self.n, ", d =", self.d, "lambda(epsilon)=", self.lambdaEpsilon)
		print("model OOV from Training Data:", self.modelOOVfromTrainingData)
		print("training set size = ", len(self.trainingData))
		print("unique entries in training set = ", self.getUniqueTrainingDataSize())
		print("observed vocabulary:")
		print(self.observedVocabulary)
		for i in range(1, self.n+1) :
			print(i, "-gram:")
			for item in self.grams[i] :
				print("\t", item, self.grams[i][item])
		print("---------------------------")
		
	def extendInputWithStartStop(self, splitInput) :
		# we shoiuld actually (actually?) add n-1 star/stop tokens
		return [self.startToken for _ in range(self.n-1)] + splitInput + [self.stopToken for _ in range(self.n-1)]
		
	def modifyInputwithUnknownEvent(self, splitInput, isTrainingMode) :
		# thing is, self.observed vocabulary is used during training and if modelOOVfromTrainingData == True,
		# i.e. we keep track of the first occurrence of symbols
		# during testing 
	
		targetInput = []
		for item in splitInput :
			if isTrainingMode :
				if self.modelOOVfromTrainingData :
					# we allow replacing the first occurrences of items with <UNK> => we do this by leveraging on self.observedVocabulary
					if item not in self.observedVocabulary :
						# OOV: change first occurrence and update observedVocabulary
						targetInput.append(self.unknownToken)
						self.observedVocabulary.append(item)
					else :
						# already observed symbol: keep it as it is
						targetInput.append(item)
				else :
					# we do not make any transformation: keep it as it is
					targetInput.append(item)
					# update observedVocabulary
					if item not in self.observedVocabulary :
						self.observedVocabulary.append(item)
			else :
				# here we replace items which are not observed in the unigrams, as observedVocabulary is just used for UNK replacement during training
				if item not in [x[0] for x in self.grams[1]] :
					# OOV item
					targetInput.append(self.unknownToken)
				else :
					targetInput.append(item)
				
		return targetInput
		
	def preprocessInputString(self, inputString, isTrainingMode, separator, printDebug) :
		
		if printDebug :
			print("** start preprocessInputString **")
		
		splitInput = inputString.split(separator)
		if printDebug :
			print("splitInput:", splitInput)
			
		splitInputWithUnknown = self.modifyInputwithUnknownEvent(splitInput, isTrainingMode)
		if printDebug :
			print("splitInputWithUnknown:", splitInputWithUnknown)
		
		splitInputExtended = self.extendInputWithStartStop(splitInputWithUnknown)
		if printDebug :
			print("splitInputExtended:", splitInputExtended)
			print("** end preprocessInputString **")
			
		return splitInputExtended
		
	def feed(self, inputString, separator='-', printDebug=False) :
		# feed should just update all the n-grams, nothing else
		
		if printDebug :
			print("** start FEED **")
			print("input string: ", inputString)
		self.trainingData.append(inputString)
		
		splitInputExtended = self.preprocessInputString(inputString, True, separator, printDebug)
		if printDebug :
			print("splitInputExtended:", splitInputExtended)
		
		# ngram update starts here
		for n in range(1, self.n+1) :
			startIndex = 0
			endIndex = len(splitInputExtended) - n
			if printDebug :
				print("updating", n, "-gramm startIndex=", startIndex, "endIndex=", endIndex)
			for i in range(startIndex, endIndex+1) :
				# get the n-gram
				gram = tuple(splitInputExtended[i : i+n])
				if printDebug :
					print(gram)
				if gram not in self.grams[n] :
					self.grams[n][gram] = 0
				self.grams[n][gram] += 1		
		
		if printDebug :
			print("** end FEED **")
			
	def indent(self, n) :
		return (self.n - n) * " "
			
	def P_KN(self, w_i, w_prev, n, whichApproach=0, printDebug=False) :
		
		if printDebug:
			print(self.indent(n), "calculate PKN(", w_i, "|", w_prev, "), n=", n)
			
		if n == 1 :
			# end of recursion:
			if printDebug :
				print(self.indent(n), "end of recursion: unigram")
			
			# extract count_w_i(w_i)
			count_w_i = 0
			if (w_i,) in self.grams[1] :
				count_w_i = self.grams[1][(w_i,)]
				
			numerator = np.max([count_w_i - self.d, 0])
			denominator = np.sum(list(self.grams[1].values()))
			
			if self.useFixedVocabularySize :
				p = float(numerator) / denominator + self.lambdaEpsilon / self.fullVocabularySize 
			else :
				p = float(numerator) / denominator + self.lambdaEpsilon / len(self.observedVocabulary)
			
			if printDebug :
				print(self.indent(n), "P_KN(", w_i, ")= max(", count_w_i,"-",self.d,",0) /", \
					denominator, "+", self.lambdaEpsilon,"/", len(self.observedVocabulary), "=", p, "[n=", n, "]")
			
			return p
			
		##### compute the first part of the recursion #####
		# max(c(w_i-n+1..i) - d, 0) / c(w_i-n+1..wi-1)
		#
		# 1. calculate c(w_i-n+1..i) and then max(count-d, 0)
		gram = tuple([w for w in w_prev] + [w_i])
		
		count_gram = 0
		if gram in self.grams[n] :
			count_gram = self.grams[n][gram]
		numerator = np.max([count_gram - self.d, 0])
		# 2. calculate sum_wi count(w_i-n+1..i-1) i.e. the count of the w_prev in the previous order gram
		denominator = 0
		if w_prev in self.grams[n-1] :
			denominator = self.grams[n-1][w_prev]
		firstPart = 0
		if denominator > 0 :
			firstPart = numerator / float(denominator)
		if printDebug :
			print(self.indent(n), "firstPart= max(", count_gram,"-", self.d, ",0)/", denominator, "=", numerator, "/", denominator, "=", firstPart)
		
		###### compute the second part of the recursion #####
		# d / sum_wi count(w_i-n+1..i) * |{w' : c(w_i-n+1..i-1,')>0}|
		# 1. calculate sum_wi count(w_i-n+1..i)
		# 2. calculate |{w' : c(w_i-n+1..i-1,')>0}| i.e. number of these ngrams with length > 0
		if printDebug :
			print(self.indent(n), "calculate second part")
		numGrams = 0
		sumGrams = 0
		for gram in self.grams[n] :
			# we should exclute the last entry in gram and compare this with wprev!
			gram_prev = gram[:-1]
			#if printDebug :
			#	print(self.indent(n), "gram=", gram, "gram_prev=", gram_prev)
			#print(self.indent(n), gram, gram_prev, w_prev, len(gram), len(gram_prev), len(w_prev), gram_prev == w_prev)
			if gram_prev == w_prev :
				numGrams += 1
				sumGrams += self.grams[n][gram]
				if printDebug :
					print(self.indent(n), "OK, count=", self.grams[n][gram], "updated sumGrams=", sumGrams, "update numGrams=", numGrams)
		secondPart = 0
		if sumGrams > 0 :
			secondPart = self.d / sumGrams * numGrams
		if printDebug :
			print(self.indent(n), "secondPart=", self.d, "/", sumGrams, "*", numGrams, "=", secondPart)
		
		###### compute the third part of the recursion: level below #####
		w_prev_recursion = w_prev[1:]
		#if printDebug :
		#	print(self.indent(n), "calculating thirdPart, w_prev=", w_prev, "w_prev_recursion=", w_prev_recursion)
		thirdPart = self.P_KN(w_i, w_prev_recursion, n-1, whichApproach, printDebug)
		if printDebug :
			print(self.indent(n), "thirdPart=", thirdPart, "w_prev_recursion=", w_prev_recursion)
		
		p = 0
		if (firstPart == 0) and (secondPart == 0) :
			if whichApproach == 0 :
				# old approach, p(w_i | hist) = p(w_i)
				p = thirdPart
				if printDebug :
					print(self.indent(n), "we cannot interpolate, we do not have enough information, therefore we backoff using just pContinuation")
					if thirdPart == 0 :
						print(self.indent(n), "ERROR! we should never have pContinuation==0 because then it means we can neither do interpolation nor backoff!")
			else :
				# new approach: p(w_i | hist) = prod_{w in hist+w_i} p(w). this interpretation leads to very strict probabilities, might be wrong?
				prodP = self.P_KN(w_i, (), 1, whichApproach)
				if printDebug :
					print(self.indent(n), "we cannot interpolate, we do not have enough information,therefore the result is the prod of independent words?")
					print("p(", w_i, ")=", prodP)
					
				for w in w_prev :
					p_w = self.P_KN(w, (), 1, whichApproach)
					prodP *= p_w
					if printDebug :
						print("p(", w, ")=", p_w, "totalProd=", prodP)
					
				p = prodP
				if p == 0 :
					if printDebug :
						print(self.indent(n), "ERROR! we should never have pContinuation==0 because then it means we can neither do interpolation nor backoff!")
			
			
		else :
			p = firstPart + secondPart * thirdPart
			if printDebug :
				print(self.indent(n), "P_KN(", w_i, "|", w_prev, ") = ", firstPart, "+", secondPart, "*", thirdPart, "=", p, "[n=", n, "]")
		
		
		return p
		
			
	def chainProbability(self, inputString, separator='-', whichApproach=0, getChainLogProb=False, printDebug=False, rounding=None) :
		# we perform the recursive KN smoothing P calculation, in theory we should be able to deal with OOV seamlessly
		splitInputExtended = self.preprocessInputString(inputString, False, separator, printDebug)
		if printDebug :
			print("** start chainProbability **")
			print("fullVocabularySize(fixed)=", self.fullVocabularySize)
			print("observedVocabulary=", self.observedVocabulary)
			print("input string: ", splitInputExtended)	

		# we traverse the string in accordance with our self.n value
		P = 0
		startIndex = 0
		endIndex = len(splitInputExtended) - self.n
		for i in range(startIndex, endIndex+1) :
			# get the n-gram
			gram = tuple(splitInputExtended[i : i + self.n])
			w_i = gram[-1]
			w_prev = gram[0:-1]
			if printDebug :
				print("\nGET P_KN(", w_i, "|", w_prev, ")")
			p = self.P_KN(w_i, w_prev, self.n, whichApproach, printDebug)
			P += np.log(p)
			if printDebug :
				print("P_KN(", w_i, "|", w_prev, ")=", p, "log=", np.log(p), "P=", P)
				print("")
		
		if not getChainLogProb :
			P = np.exp(P)
		
		if printDebug :
			print("finally P=", P)
			print("** end chainProbability **")
		
		if rounding != None :
			return np.round(P, rounding)
		else :
			return P
	
	def walk(self, addStartStopToken=False, printDebug=False) :
		# walking will be based on the highest gram
		generatedSequence = []
		
		# we have ngrams now!		
		curNode = tuple([self.startToken for _ in range(self.n-1)])
		if printDebug :
			print("starting to walk from curNode=", curNode)
		
		while curNode[-1] != self.stopToken :
			if curNode[-1] == self.startToken :
				if addStartStopToken :
					generatedSequence.append(curNode[-1])
			else :
				generatedSequence.append(curNode[-1])
				
			if printDebug :
				print("onto the loop, curNode=", curNode)
				print("current generatedSequence", generatedSequence)
			dicto = self.getGramDict(curNode, printDebug)
			if printDebug :
				print("dicto to sapkle:", dicto)
			sampledNode = self.sample(dicto, printDebug=printDebug)
			if printDebug :
				print("I sampled", sampledNode)
			# generate a new curNode: trim the [0] entry of curNode and append sampledNode
			curNode = tuple(list(curNode)[1:] + [sampledNode])
			
		# we need to add the stop token
		if addStartStopToken :
			generatedSequence.append(curNode[-1])
		
		return generatedSequence
		
	def getGramDict(self, fromNode, printDebug=False) :
		gramDict = {}
		for gram in self.grams[self.n] :
			if gram[:len(fromNode)] == fromNode :
				if printDebug :
					print(gram)
				gramDict[gram[-1]] = self.grams[self.n][gram]
		return gramDict
		
	def sample(self, inputDict, sampleSize=1, printDebug=False) :
		# probabilistic sampling on smoothed probabilities
		items = list(inputDict.keys())
		p = np.asarray(list(inputDict.values()))
		p = p / np.sum(p)
		sampledItem = np.random.choice(items, p=p)
		if sampleSize != 1 :
			repl = False if sampleSize < len(items) else True
			sampledItem = np.random.choice(items, size=sampleSize, replace=repl, p=p)
		
		if printDebug :
			print("sampling", sampleSize, "items. input:", inputDict)
			print("keys=", items)
			print("values=", inputDict.values())
			print("p=", p)
			print(sampledItem)
		return sampledItem
	
	def printTODO(self) :
		#print("find out about beginnings and ends, do we need them or what?")
		print("we must find out whether our patch on chain probability, the whichApprach so to speak, is correct or make sense")
		print("render function should be made a bit more efficient, it duplicates its code 3 times!")
			
	def render(self,
	
		labelDF = None,
		labelDF_from = None,
		labelDF_to = None,

		nodeShape = "oval",
		nodeStyle = "filled",
		nodeFillColor = "#9ACEEB",

		startNodeLabel = "start",
		startNodeShape = "square",
		startNodeStyle = "filled",
		startNodeFillColor = "#2bb217",

		stopNodeLabel = "stop",
		stopNodeShape = "square",
		stopNodeStyle = "filled",
		stopNodeFillColor = "#2bb217",

		showUnknownNode = False,
		unknownNodeShape = "diamond",
		unknownNodeStyle = "filled",
		unknownNodeFillColor = "#FCD975",
		
		edgeOffset = 1.0,
		edgeScaleFactor = 2.0,
		edgeWeightStyle = "continuous",
		wMultiplierList = [0.1, 0.5],
		wColours = [(0x11,0x11,0x11,0x11),(0xaa,0xaa,0xaa,0xaa),(0,0,0,0xff)],
		
		roundingDecimals=2,

		renderEdgeWeightThreshold = 0,   # below this we do not show the edge
		renderEdgeLabel = True,
		renderEdgeAlpha = True,
		
		fontName = "monospace",

		printDebug = False

		) :

		#if(self.n != 2) :
		#	print("HA! ngram has n=", self.n, ", we cannot really generate a meaningful graph representation, for now we just show bigrams!")
		
		dot = Digraph(comment=self.name)
		dot.graph_attr['label'] = self.name + " (D=" + str(len(self.trainingData)) + ", U=" + str(len(set(self.trainingData))) + ")"
		dot.graph_attr['labelloc'] = "top"
		dot.graph_attr['labeljust'] = "left"
		
		# prepare the node list: start with <startNodeLabel>
		dot.node(startNodeLabel, shape=startNodeShape, style=startNodeStyle, fontname=fontName,	fillcolor=startNodeFillColor)
		
		# add the beginning list
		renderedNodeList = []
		renderedNodeList.append(startNodeLabel)
		
		for bigram in self.grams[2] :
			if bigram[0] != self.startToken :
				continue
			
			if bigram[1] == self.startToken :
				continue
				
			nodeStr = bigram[1]
			
			if labelDF is not None :
				# we might have to do a typecast: nodeStr is a string, but the value in labelDF_from might not. so in case we change the type of nodeStr
				targetValue = nodeStr
				if labelDF[labelDF_from].dtype != str :
					targetValue = int(nodeStr)
				nodeStr = labelDF.loc[labelDF[labelDF_from] == targetValue][labelDF_to].tolist()[0]
			
			#print(nodeStr)
			#print(len(nodeStr))
			
			# we might have "unknown" in the beginning list, we might want to treat him with good care

			# do we render the start - beginning node+edge?
			edgeWeight = self.P_KN((bigram[1]), (bigram[0],), n=2, whichApproach=0)#, printDebug=True)
			#if edgeWeight < renderEdgeWeightThreshold :
			#	continue

			if (self.unknownToken in nodeStr) and (not showUnknownNode) :
				if printDebug : print("skip unknown")
				continue

			# we can render!
			renderedNodeList.append(nodeStr)
			
			#nodeStr = nodeStr.replace(" ", "_")

			if (self.unknownToken in nodeStr) :
				dot.node(nodeStr, shape=unknownNodeShape, style=unknownNodeStyle, fillcolor=unknownNodeFillColor, fontname=fontName)
			else : 
				dot.node(nodeStr, shape=nodeShape, style=nodeStyle, fillcolor=nodeFillColor, fontname=fontName)
				
			# now we add an edge from start to the newly added node
			wScaled = edgeOffset + edgeScaleFactor * edgeWeight

			#if edgeWeight >= renderEdgeWeightThreshold :
			if edgeWeightStyle == "continuous" :
				if renderEdgeAlpha :
					# the width is meaningless in this case
					a = int(255 * (edgeWeight))
					h = "#%02x%02x%02x%02x" % (0, 0, 0, a)
					if renderEdgeLabel :
						dot.edge(startNodeLabel, nodeStr, color=h, label=str(np.round(edgeWeight, roundingDecimals)))
					else :
						dot.edge(startNodeLabel, nodeStr, color=h)
				else :
					# we render the width, the colour is meaningless
					if renderEdgeLabel :
						dot.edge(startNodeLabel, nodeStr, penwidth=str(np.round(wScaled, roundingDecimals)), label=str(np.round(edgeWeight, roundingDecimals)))
					else :
						dot.edge(startNodeLabel, nodeStr, penwidth=str(np.round(wScaled, roundingDecimals)))
			elif edgeWeightStyle == "3types" :
				# 3 thicknesses
				#wMultiplier = 0
				
				c = "#%02x%02x%02x%02x" % (0, 0, 0, 0)
				#print(c)
				if edgeWeight < wMultiplierList[0] :
					c = wColours[0]
					wMultiplier = 0.66
				elif edgeWeight < wMultiplierList[1] :
					c = wColours[1]
					wMultiplier = 0.33
				else :
					c = wColours[2]
					wMultiplier = 0
					
				#print(c)
					
				#a = int(255 * wMultiplier)
				h = "#%02x%02x%02x%02x" % c# (a, a, a, 255)
				
				#print(h)
					
				if renderEdgeLabel :
					dot.edge(startNodeLabel, nodeStr, color=h, penwidth=str(np.round(wScaled, roundingDecimals)), label=str(np.round(edgeWeight, roundingDecimals)))
				else :
					dot.edge(startNodeLabel, nodeStr, color=h, penwidth=str(np.round(wScaled, roundingDecimals)))
					
			elif edgeWeightStyle == "1type" :
				c = "#%02x%02x%02x%02x" % (0x00,0x00,0x00,0xff)
				wScaled = 1
				
				
				#print(h)
					
				if renderEdgeLabel :
					dot.edge(startNodeLabel, nodeStr, color=c, penwidth=str(np.round(wScaled, roundingDecimals)), label=str(np.round(edgeWeight, roundingDecimals)))
				else :
					dot.edge(startNodeLabel, nodeStr, color=c, penwidth=str(np.round(wScaled, roundingDecimals)))
			
			else :
				print("unknown edgeWeightStyle", edgeWeightStyle, "current possibilities are either continuous, 1type or 3types")
				
		
		# now the real ngram
		for bigram in self.grams[2] :
			if (bigram[0] == self.startToken) or \
				(bigram[0] == self.stopToken) or \
				(bigram[1] == self.startToken) or \
				(bigram[1] == self.stopToken) :
				continue
					
			fromNodeStr = bigram[0]
			toNodeStr = bigram[1]
			
			edgeWeight = self.P_KN((toNodeStr), (fromNodeStr, ), n=2)
			
			if labelDF is not None :
				# we might have to do a typecast: nodeStr is a string, but the value in labelDF_from might not. so in case we change the type of nodeStr
				targetValueFromNode = fromNodeStr
				targetValueToNode = toNodeStr
				if labelDF[labelDF_from].dtype != str :
					targetValueFromNode = int(fromNodeStr)
					targetValueToNode = int(toNodeStr)
				
				fromNodeStr = labelDF.loc[labelDF[labelDF_from] == targetValueFromNode][labelDF_to].tolist()[0]
				toNodeStr = labelDF.loc[labelDF[labelDF_from] == targetValueToNode][labelDF_to].tolist()[0]
			
			#print(fromNodeStr, toNodeStr, edgeWeight)

			# we can render the edge, just make sure we are not dealing with "unknown" neither at from or to
			if ((self.unknownToken in fromNodeStr) or (self.unknownToken in toNodeStr)) and (not showUnknownNode) :
				# it is unknown but we dont want it, ignore!
				continue

			# we can render the node as well!
			if fromNodeStr not in renderedNodeList :
				# render fromNode
				if (self.unknownToken in fromNodeStr) :
					dot.node(fromNodeStr, shape=unknownNodeShape, style=unknownNodeStyle, fillcolor=unknownNodeFillColor, fontname=fontName)
				else : 
					dot.node(fromNodeStr, shape=nodeShape, style=nodeStyle, fillcolor=nodeFillColor, fontname=fontName)

			if toNodeStr not in renderedNodeList :
				# render toNode
				if (self.unknownToken in toNodeStr) :
					dot.node(toNodeStr, shape=unknownNodeShape, style=unknownNodeStyle, fillcolor=unknownNodeFillColor, fontname=fontName)
				else : 
					dot.node(toNodeStr, shape=nodeShape, style=nodeStyle, fillcolor=nodeFillColor, fontname=fontName)

			# we can render the edge!
			if edgeWeightStyle == "continuous" :
				if edgeWeight >= renderEdgeWeightThreshold :
					if renderEdgeAlpha :
						# the width is meaningless in this case
						a = int(255 * (edgeWeight))
						h = "#%02x%02x%02x%02x" % (0, 0, 0, a)
						if renderEdgeLabel :
							dot.edge(fromNodeStr, toNodeStr, color=h, label=str(np.round(edgeWeight, roundingDecimals)))
						else :
							dot.edge(fromNodeStr, toNodeStr, color=h)

					else :
						# we render the width, the colour is meaningless
						if renderEdgeLabel :
							dot.edge(fromNodeStr, toNodeStr, penwidth=str(np.round(wScaled, roundingDecimals)), label=str(np.round(edgeWeight, roundingDecimals)))
						else :
							dot.edge(fromNodeStr, toNodeStr, penwidth=str(np.round(wScaled, roundingDecimals)))
			
			elif edgeWeightStyle == "3types" :
			
				# 3 thicknesses
				c = "#%02x%02x%02x%02x" % (0, 0, 0, 0)
				#print(c)
				if edgeWeight < wMultiplierList[0] :
					c = wColours[0]
					wMultiplier = 0.66
				elif edgeWeight < wMultiplierList[1] :
					c = wColours[1]
					wMultiplier = 0.33
				else :
					c = wColours[2]
					wMultiplier = 0
					
				#print(c)
					
				#a = int(255 * wMultiplier)
				h = "#%02x%02x%02x%02x" % c# (a, a, a, 255)
					
				if renderEdgeLabel :
					dot.edge(fromNodeStr, toNodeStr, penwidth=str(np.round(wScaled, roundingDecimals)), color=h, label=str(np.round(edgeWeight, roundingDecimals)))
				else :
					dot.edge(fromNodeStr, toNodeStr, color=h, penwidth=str(np.round(wScaled, roundingDecimals)))
			
			elif edgeWeightStyle == "1type" :
				c = "#%02x%02x%02x%02x" % (0x00,0x00,0x00,0xff)
				wScaled = 1
				
				if renderEdgeLabel :
					dot.edge(fromNodeStr, toNodeStr, penwidth=str(np.round(wScaled, roundingDecimals)), color=c, label=str(np.round(edgeWeight, roundingDecimals)))
				else :
					dot.edge(fromNodeStr, toNodeStr, color=c, penwidth=str(np.round(wScaled, roundingDecimals)))
							
			else :
				print("unknown edgeWeightStyle", edgeWeightStyle, "current possibilities are either continuous, 1type or 3types")
				
			#break
				
		# finalise the node list: add <stopNodeLabel> after the ends
		dot.node(stopNodeLabel, shape=stopNodeShape, style=stopNodeStyle, fillcolor=stopNodeFillColor, fontname=fontName)
		
		for bigram in self.grams[2] :
			if bigram[1] != self.stopToken :
				continue
				
			if bigram[0] == self.stopToken :
				continue
		
			nodeStr = bigram[0]
			
			# do we render the "end node" -> stop edge?
			edgeWeight = self.P_KN((bigram[1]), (bigram[0], ), n=2)
			#if edgeWeight < renderEdgeWeightThreshold :
			#	continue
			
			if labelDF is not None :
				# we might have to do a typecast: nodeStr is a string, but the value in labelDF_from might not. so in case we change the type of nodeStr
				targetValue = nodeStr
				if labelDF[labelDF_from].dtype != str :
					targetValue = int(nodeStr)
				nodeStr = labelDF.loc[labelDF[labelDF_from] == targetValue][labelDF_to].tolist()[0]
				

			if (self.unknownToken in nodeStr) and (not showUnknownNode) :
				if printDebug : print("skip unknown")
				continue

			# we can render!
			renderedNodeList.append(nodeStr)

			if (self.unknownToken in nodeStr) :
				dot.node(nodeStr, shape=unknownNodeShape, style=unknownNodeStyle, fillcolor=unknownNodeFillColor, fontname=fontName)
			else : 
				dot.node(nodeStr, shape=nodeShape, style=nodeStyle, fillcolor=nodeFillColor, fontname=fontName)

			# now we add an edge from the node to stop
			wScaled = edgeOffset + edgeScaleFactor * edgeWeight

			if edgeWeightStyle == "continuous" :
				#if edgeWeight >= renderEdgeWeightThreshold :
				if renderEdgeAlpha :
					# the width is meaningless in this case
					a = int(255 * (edgeWeight))
					h = "#%02x%02x%02x%02x" % (0, 0, 0, a)
					if renderEdgeLabel :
						dot.edge(nodeStr, stopNodeLabel, color=h, label=str(np.round(edgeWeight, roundingDecimals)))
					else :
						dot.edge(nodeStr, stopNodeLabel, color=h)
				else :
					if renderEdgeLabel :
						dot.edge(nodeStr, stopNodeLabel, penwidth=str(np.round(wScaled, roundingDecimals)), label=str(np.round(edgeWeight, roundingDecimals)))
					else :
						dot.edge(nodeStr, stopNodeLabel, penwidth=str(np.round(wScaled, roundingDecimals)))
			elif edgeWeightStyle == "3types" :
				# 3 thicknesses
				c = "#%02x%02x%02x%02x" % (0, 0, 0, 0)
				#print(c)
				if edgeWeight < wMultiplierList[0] :
					c = wColours[0]
					wMultiplier = 0.66
				elif edgeWeight < wMultiplierList[1] :
					c = wColours[1]
					wMultiplier = 0.33
				else :
					c = wColours[2]
					wMultiplier = 0
					
				#print(c)
					
				#a = int(255 * wMultiplier)
				h = "#%02x%02x%02x%02x" % c# (a, a, a, 255)
					
				if renderEdgeLabel :
					dot.edge(nodeStr, stopNodeLabel, penwidth=str(np.round(wScaled, roundingDecimals)), color=h, label=str(np.round(edgeWeight, roundingDecimals)))
				else :
					dot.edge(nodeStr, stopNodeLabel, color=h, penwidth=str(np.round(wScaled, roundingDecimals)))
			elif edgeWeightStyle == "1type" :
				c = "#%02x%02x%02x%02x" % (0x00,0x00,0x00,0xff)
				wScaled = 1
				
				
				#print(h)
					
				if renderEdgeLabel :
					dot.edge(nodeStr, stopNodeLabel, penwidth=str(np.round(wScaled, roundingDecimals)), color=c, label=str(np.round(edgeWeight, roundingDecimals)))
				else :
					dot.edge(nodeStr, stopNodeLabel, color=c, penwidth=str(np.round(wScaled, roundingDecimals)))
					
			else :
				print("unknown edgeWeightStyle", edgeWeightStyle, "current possibilities are either continuous, 1type or 3types")
			
						
		return dot
	