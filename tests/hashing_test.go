package tests

import (
	"proxycache"
	"reflect"
	"testing"
)

func TestSanitizeBackendDir(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"10.0.0.1:8000", "10.0.0.1-8000"},
		{"127.0.0.1:8080", "127.0.0.1-8080"},
		{"host:8000:extra", "host-8000-extra"},
		{"no-colons", "no-colons"},
		{"", ""},
	}
	for _, c := range cases {
		if got := proxycache.SanitizeBackendDir(c.in); got != c.want {
			t.Errorf("SanitizeBackendDir(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestBlockHashesFromTokens(t *testing.T) {
	cases := []struct {
		tokens []int
		wpb    int
		want   []string
	}{
		{[]int{1, 2, 3}, 3, []string{"8a6ae15122001229edb8866f56e342af12ae8187203c3e3b33931743e7c0c48d"}},
		{[]int{1, 2, 3, 4, 5}, 3, []string{
			"8a6ae15122001229edb8866f56e342af12ae8187203c3e3b33931743e7c0c48d",
			"c73bcaadd94985269eeafd457c9f395135874dad5536cf1f6d75c132f602a14c",
		}},
		{[]int{1, 2, 3, 4}, 2, []string{
			"17f8af97ad4a7f7639a4c9171d5185cbafb85462877a4746c21bdb0a4f940ca0",
			"5cc3ec012284b76c9ebb137c692826dcba12e4cc6792de419b51667b02df4b58",
		}},
		{[]int{1}, 100, []string{"6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b"}},
		{[]int{}, 100, []string{}},
		{nil, 100, []string{}},
	}
	for _, c := range cases {
		got := proxycache.BlockHashesFromTokens(c.tokens, c.wpb)
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("BlockHashesFromTokens(%v, %d) = %v, want %v", c.tokens, c.wpb, got, c.want)
		}
	}
}

func TestBlockHashesFromTokensDefault(t *testing.T) {
	if proxycache.WordsPerBlock != 100 {
		t.Skipf("WORDS_PER_BLOCK=%d, expected default 100", proxycache.WordsPerBlock)
	}
	tokens := []int{7, 8, 9}
	want := proxycache.BlockHashesFromTokens(tokens, 100)
	if got := proxycache.BlockHashesFromTokensDefault(tokens); !reflect.DeepEqual(got, want) {
		t.Errorf("BlockHashesFromTokensDefault(%v) = %v, want %v", tokens, got, want)
	}
}

func TestLCPBlocks(t *testing.T) {
	a := []string{"x", "y", "z"}
	b := []string{"x", "y", "q"}
	cases := []struct {
		b1, b2 []string
		want   int
	}{
		{a, a, 3},
		{a, b, 2},
		{[]string{"x"}, a, 1},
		{a, []string{"x", "y", "z", "w"}, 3},
		{[]string{"q"}, a, 0},
		{a, nil, 0},
		{nil, nil, 0},
	}
	for _, c := range cases {
		if got := proxycache.LCPBlocks(c.b1, c.b2); got != c.want {
			t.Errorf("LCPBlocks(%v, %v) = %d, want %d", c.b1, c.b2, got, c.want)
		}
	}
}

func TestPrefixKeySha256(t *testing.T) {
	if got := proxycache.PrefixKeySha256("hello"); got != "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824" {
		t.Errorf("PrefixKeySha256(\"hello\") = %s", got)
	}
}

func TestMetaKey(t *testing.T) {
	cases := []struct {
		name   string
		tokens []int
		want   string
	}{
		{"ModelA", []int{1, 2, 3}, "80fba1fef05cba2f90b34c833dba92fc2abc63ce264d61d1eebf24a92169f025"},
		{"ModelB", []int{7, 8}, "1cc62576f41245b06f21c4143e5cecf790d82aa317be298f59b89ac75579ff98"},
	}
	for _, c := range cases {
		if got := proxycache.MetaKey(c.name, c.tokens); got != c.want {
			t.Errorf("MetaKey(%q, %v) = %s, want %s", c.name, c.tokens, got, c.want)
		}
	}
}
